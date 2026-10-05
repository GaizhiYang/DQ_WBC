"""Real two-process Gloo integration for synchronous PPO without Isaac Gym."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))
from utils import asymmetric_distributed as distributed
from utils.asymmetric_teacher_training import make_agent, AsymmetricTeacherTrainingState, restore_experiment
from utils.asymmetric_teacher_preprocessor import export_deployment, load_deployment
from modules.asymmetric_teacher import PROPRIO_INDICES
from learning.asymmetric_teacher_trainer import AsymmetricTeacherTrainer
from test_asymmetric_teacher_training import SyntheticFeatureWrapper
from test_teacher_vision_training import ReusingCameraEnv
from utils.config import get_params


class RankEnvironment(ReusingCameraEnv):
    def _observe(self):
        result = super()._observe()
        result["obs"] += distributed.rank() * 0.5
        return result


def assert_replicated(tensor):
    reference = tensor.detach().clone()
    dist.broadcast(reference, 0)
    torch.testing.assert_close(tensor, reference, atol=0, rtol=0)


def distributed_worker(worker_rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(directory) / "rendezvous"),
                            rank=worker_rank, world_size=2, timeout=timedelta(seconds=90))
    agents = []
    try:
        torch.manual_seed(100 + worker_rank)
        settings = dict(mode="m2", rollouts=2, minibatch_size=2, learning_epochs=2,
                        learning_rate=1e-4, total_steps=4, eval_interval=2, eval_steps=3,
                        checkpoint_interval=2, seed=100, initialization="scratch",
                        normalization="rollout", world_size=2)

        def build():
            env = SyntheticFeatureWrapper(RankEnvironment(0), m1=True, m2=True)
            agent = make_agent(env, settings, directory, "shared")
            agents.append(agent)
            agent.checkpoint_modules["asymmetric_teacher_state"] = AsymmetricTeacherTrainingState(
                env, {"asymmetric_teacher": settings}, {}, 0)
            return env, agent

        env, agent = build()
        distributed.synchronize_agent(agent)
        before = agent.policy.actor.visual_adapter.weight.detach().clone()
        for model in (agent.policy, agent.value):
            for parameter in model.parameters():
                assert_replicated(parameter)
        # Check gradient averaging independently from PPO and before clipping.
        for parameter in agent._parameters:
            parameter.grad = torch.full_like(parameter, float(1 + 2 * worker_rank))
        agent._average_gradients()
        assert all(torch.equal(p.grad, torch.full_like(p, 2.)) for p in agent._parameters)
        agent.optimizer.zero_grad(set_to_none=True)

        def trainer(env, agent, steps):
            return AsymmetricTeacherTrainer(env, agent, cfg=dict(
                timesteps=steps, headless=True, disable_progressbar=True,
                close_environment_at_exit=False, eval_interval=2, eval_steps=3,
                checkpoint_interval=2, eval_seed=1729 + worker_rank))

        first = trainer(env, agent, 2)
        first.train()
        assert not torch.equal(before, agent.policy.actor.visual_adapter.weight)
        assert int(agent._state_preprocessor.current_count) == 9
        # Global statistics equal a centralized batch plus the count=1 prior.
        local = agent.memory.get_tensor_by_name("states")[..., :1276].reshape(-1, 1276).double()
        gathered = [torch.empty_like(local) for _ in range(2)]
        dist.all_gather(gathered, local)
        samples = torch.cat(gathered)
        expected_mean = samples.sum(0) / 9
        expected_variance = (samples.square().sum(0) + 1) / 9 - expected_mean.square()
        torch.testing.assert_close(agent._state_preprocessor.running_mean, expected_mean)
        torch.testing.assert_close(agent._state_preprocessor.running_variance, expected_variance)
        assert first.last_evaluation["transitions"] == 12
        assert first.last_evaluation["world_size"] == 2
        assert agent.write_interval == (2 if worker_rank == 0 else 0)
        if worker_rank:
            assert not hasattr(agent, "writer")
            assert not agent.tracking_data

        for module in (agent.policy, agent.value, agent._state_preprocessor):
            for tensor in module.state_dict().values():
                assert_replicated(tensor)
        for state in agent.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    assert_replicated(value)
        assert_replicated(torch.tensor(agent.optimizer.param_groups[0]["lr"]))
        dist.barrier()
        saved = torch.load(Path(directory) / "shared/checkpoints/agent_2.pt", weights_only=False)
        next_env, resumed = build()
        restore_experiment(resumed, saved)
        distributed.synchronize_agent(resumed)
        trainer(next_env, resumed, 4).train()
        assert int(resumed._state_preprocessor.current_count) == 17
        assert resumed.checkpoint_modules["asymmetric_teacher_state"].completed_steps == 4
        for module in (resumed.policy, resumed.value, resumed._state_preprocessor):
            for tensor in module.state_dict().values():
                assert_replicated(tensor)
        states = next_env.reset()[0]
        deployed = load_deployment(export_deployment(resumed.policy, resumed._state_preprocessor))
        with torch.no_grad():
            expected = resumed.policy.compute({"states": resumed._state_preprocessor(states)}, "policy")[0]
            actual = deployed(states[:, 1276:63484], states[:, list(PROPRIO_INDICES)], states[:, -16:])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        # Unequal numbers of completed episodes must not produce mean-of-ratios.
        count = 1 if worker_rank == 0 else 3
        metric = dict(vector_steps=5, transitions=10, completed_episodes=count,
                      successes=1 if worker_rank == 0 else 0,
                      gsr=1. if worker_rank == 0 else 0., ossr=1. if worker_rank == 0 else 0.,
                      mean_reward=worker_rank + 1., mean_episode_return=worker_rank * 10.,
                      tsc=4. if worker_rank == 0 else None, initial_base_x_range=[worker_rank, worker_rank + 1],
                      perception={"prediction_error_m": .2 if worker_rank == 0 else .8})
        combined = distributed.aggregate_evaluation(metric, {"prediction_error_m": count})
        assert combined["gsr"] == .25 and combined["ossr"] == .25
        assert combined["mean_episode_return"] == 7.5 and combined["mean_reward"] == 1.5
        assert combined["tsc"] == 4. and abs(combined["perception"]["prediction_error_m"] - .65) < 1e-12
        assert combined["initial_base_x_range"] == [0, 2]
        dist.barrier()
        if worker_rank == 0:
            records = (Path(directory) / "shared/evaluations.jsonl").read_text().splitlines()
            assert [json.loads(line)["completed_steps"] for line in records] == [2, 4]
    finally:
        for agent in agents:
            if hasattr(agent, "writer"):
                agent.writer.close()
        dist.destroy_process_group()


class AsymmetricDistributedTests(unittest.TestCase):
    def test_two_rank_real_ppo_collectives_resume_evaluation_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(directory,), nprocs=2, join=True)

    def test_torchrun_devices_and_explicit_vulkan_mapping(self):
        variables = {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "LOCAL_RANK": "1", "RANK": "1"}
        with mock.patch.dict(os.environ, variables), mock.patch.object(sys, "argv", ["asymmetric"]):
            args = get_params()
            distributed.configure_devices(args)
            self.assertEqual((args.sim_device, args.rl_device, args.graphics_device_id), ("cuda:1", "cuda:1", 1))
        with mock.patch.dict(os.environ, variables), mock.patch.object(sys, "argv", ["asymmetric", "--graphics_device_ids", "3,2"]):
            args = get_params()
            distributed.configure_devices(args)
            self.assertEqual(args.graphics_device_id, 2)
        for flags in (["--graphics_device_ids", "0"], ["--graphics_device_ids", "0,0"],
                      ["--graphics_device_id", "0"], ["--sim_device", "cuda:1"]):
            with self.subTest(flags=flags), mock.patch.dict(os.environ, variables), mock.patch.object(sys, "argv", ["asymmetric"] + flags):
                with self.assertRaises(ValueError):
                    distributed.configure_devices(get_params())


if __name__ == "__main__":
    unittest.main()
