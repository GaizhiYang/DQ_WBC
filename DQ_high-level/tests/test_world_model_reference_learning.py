"""Pure reference scheduling, fixed exploration, normalization and BPTT ownership."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.wm_reference import ReferenceSchedule, ReferenceTeacherAgent
from utils.world_model.reference_recipe import REFERENCE_CONFIG
from utils.world_model.reference_policy import ReferenceTeacherPolicy, EmpiricalObservationNorm
from utils.world_model.imagination import ImaginationConfig, ImaginationTrainer, imagination_objective
from utils.world_model.gradients import freeze_parameters
from test_world_model_imagined_critic import Actor, Adapter, Model


class ReferenceLearningTests(unittest.TestCase):
    def test_source_collection_is_high_step_2_3_and_warmup_has_no_update_debt(self):
        config = {**REFERENCE_CONFIG, "collection_warmup_high_steps": 4,
                  "world_model_only_iterations": 2, "batch_size": 16}
        schedule = ReferenceSchedule(config)
        boundaries, actors = [], []
        for high in range(1, 15):
            valid = high*8*2
            if schedule.ready(high, valid):
                plan = schedule.plan(valid)
                boundaries.append(high)
                actors.append(plan["actor"])
                self.assertEqual(plan["new_valid_low"], (2 if len(boundaries)%2 else 3)*16)
                schedule.advance()
        self.assertEqual(boundaries, [6, 9, 11, 14])
        self.assertEqual(actors, [False, False, True, True])

    def test_explicit_model_cap_logs_discarded_integer_credit(self):
        config = {**REFERENCE_CONFIG, "batch_size": 8, "wm_max_batches": 2}
        schedule = ReferenceSchedule(config)
        result = schedule.plan(17)
        self.assertEqual(result["model_batches"], 2)
        self.assertEqual(result["discarded_batches"], 6)
        self.assertEqual(schedule.model_fraction, .5)

    def test_actor_normalizer_matches_rsl_empirical_moments_without_forward_mutation(self):
        norm = EmpiricalObservationNorm(3)
        x, y = torch.randn(7, 3), torch.randn(11, 3)*2+5
        norm.update(x)
        norm.update(y)
        combined = torch.cat((x,y))
        torch.testing.assert_close(norm._mean[0], combined.mean(0))
        torch.testing.assert_close(norm._var[0], combined.var(0, unbiased=False))
        before = deepcopy(norm.state_dict())
        z = torch.randn(3,3,requires_grad=True)
        norm(z).sum().backward()
        self.assertGreater(float(z.grad.abs().sum()), 0)
        for key, value in norm.state_dict().items():
            torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_policy_is_flat_deterministic_and_fixed_noise_has_no_trainable_std(self):
        policy = ReferenceTeacherPolicy(5,9,"cpu",hidden=(8,4),amp="none")
        observation = torch.randn(10,5)
        self.assertNotIn("log_std_parameter",dict(policy.named_parameters()))
        first = policy.act({"states":observation})[0]
        torch.testing.assert_close(first,policy.act({"states":observation})[0])
        self.assertFalse(torch.equal(first,policy.act({"states":observation,"explore":True})[0]))
        self.assertTrue(bool(((first[:,6]>0)&(first[:,6]<1)).all()))
        self.assertEqual(int(policy.obs_norm.count),0)
        policy.observe_real_next(observation)
        self.assertEqual(int(policy.obs_norm.count),10)

    def test_reference_actor_precedes_critic_normalization_and_fixed_targets(self):
        actor = Actor()
        trainer = ImaginationTrainer(Model(),Adapter(),actor,None,ImaginationConfig(
            reference_mode=True,value_source="imagined",horizon=3,lr=.0002,strength=1.,
            kl_rollback=False,critic_hidden=(8,),critic_running_norm=True,critic_activation="elu",
            objective_reduction="sum",critic_epochs=2,critic_minibatches=2,chunk_size=2))
        state = torch.tensor([[.5,0.],[1.,0.],[1.5,0.]])
        context = torch.ones(3,1)
        events = []
        actor_step = trainer.optimizer.step
        norm_update = trainer.imagined_critic.norm.update
        with patch("utils.world_model.imagination.diagonal_gaussian_kl",side_effect=AssertionError("No Gaussian KL in pure BPTT")), \
             patch("utils.world_model.imagination.deepcopy",side_effect=AssertionError("No legacy Adam snapshots in pure BPTT")), \
             patch.object(trainer.optimizer,"step",side_effect=lambda *a,**k:(events.append("actor"),actor_step(*a,**k))[1]), \
             patch.object(trainer.imagined_critic.norm,"update",side_effect=lambda x:(events.append("norm"),norm_update(x))[1]):
            result = trainer.update(state,context)
        self.assertEqual(events,["actor","norm"])
        self.assertEqual(int(trainer.imagined_critic.norm.count),9)
        self.assertEqual(result["critic_updates"],4)
        self.assertIsNone(actor.log_std.grad)

    def test_forward_hard_done_retains_backward_survival_gradient_once(self):
        class STAdapter(Adapter):
            def task_outcome(self,previous,state,action,context):
                result=super().task_outcome(previous,state,action,context)
                smooth=state[:,0].sigmoid()
                result["imagined_done"]=torch.zeros_like(smooth)+smooth-smooth.detach()
                return result
        actor=Actor()
        trainer=ImaginationTrainer(Model(),STAdapter(),actor,None,ImaginationConfig(
            reference_mode=True,value_source="imagined",horizon=2,lr=.0002,strength=1.,
            kl_rollback=False,critic_hidden=(),use_tail_value=False))
        state=torch.tensor([[.5,0.]])
        with freeze_parameters(trainer.model,trainer.imagined_critic):
            rollout=trainer.imagine(state,torch.ones(1,1))
            objective=imagination_objective(rollout,.99)
            objective.backward()
        torch.testing.assert_close(rollout.survival,torch.ones_like(rollout.survival))
        self.assertGreater(abs(float(actor.weight.grad[0])),0)

    def test_reference_sum_objective_has_source_gradient_without_horizon_divisor(self):
        def make(reduction):
            torch.manual_seed(9)
            return ImaginationTrainer(Model(),Adapter(),Actor(),None,ImaginationConfig(
                reference_mode=True,value_source="imagined",horizon=3,lr=.0002,strength=1.,
                kl_rollback=False,critic_hidden=(),objective_reduction=reduction,
                critic_epochs=1,critic_minibatches=1))
        summed, averaged = make("sum"), make("mean")
        state=torch.tensor([[.5,0.],[1.,0.]])
        a=summed.update(state,torch.ones(2,1))
        b=averaged.update(state,torch.ones(2,1))
        self.assertAlmostEqual(a["actor_grad_norm"],3*b["actor_grad_norm"],places=5)

    def test_st_first_terminal_gradient_matches_source_cumprod_without_absorption(self):
        class Endpoint(Adapter):
            def reward(self,previous,state,action,context):
                return 1+state[:,0]
            def task_outcome(self,previous,state,action,context):
                first=state[:,1]==1
                soft=state[:,0].sigmoid()*first
                done=first.float()+(soft-soft.detach())
                zero=torch.zeros_like(first)
                return {"failed":zero,"completed":first,"timeout":zero,"imagined_done":done}
        adapter=Endpoint()
        class TaskModel(Model):
            def __init__(self):
                super().__init__()
                self.active_calls=[]
            def step_with_task(self,state,action,context,dt,active=None):
                self.active_calls.append(active)
                result=self.step(state,action,context,dt)
                if active is not None:
                    result=torch.where(active[:,None],result,state)
                return result,adapter.reward(state,result,action,context),adapter.task_outcome(state,result,action,context)
        trainer=ImaginationTrainer(TaskModel(),adapter,Actor(),None,ImaginationConfig(
            reference_mode=True,value_source="imagined",horizon=3,lr=.0002,strength=1.,
            kl_rollback=False,critic_hidden=(),use_tail_value=False))
        initial=torch.tensor([[.5,0.],[1.,0.]])
        context=torch.ones(2,1)
        with freeze_parameters(trainer.model,trainer.imagined_critic):
            rollout=trainer.imagine(initial,context)
            got=imagination_objective(rollout,.99)
            actual=torch.autograd.grad(got,(trainer.actor.weight,trainer.actor.bias))
            state=initial
            rewards,dones=[],[]
            for _ in range(3):
                mean,_=trainer._policy(adapter.observation(state,context))
                action=trainer._action(mean)
                next_state=trainer.model.step(state,action,context,.15)
                rewards.append(adapter.reward(state,next_state,action,context))
                dones.append(adapter.task_outcome(state,next_state,action,context)["imagined_done"])
                state=next_state
            alive=torch.cumprod(1-torch.stack(dones),0)
            before=torch.cat((torch.ones_like(alive[:1]),alive[:-1]),0)
            expected=(.99**torch.arange(3)[:,None]*before*torch.stack(rewards)).sum(0).mean()
            reference=torch.autograd.grad(expected,(trainer.actor.weight,trainer.actor.bias))
        self.assertTrue(all(active is None for active in trainer.model.active_calls))
        self.assertTrue(bool((rollout.rewards[1:]>0).all()))
        self.assertFalse(torch.equal(rollout.states[1],rollout.states[-1]))
        torch.testing.assert_close(got,expected)
        for actual_gradient,reference_gradient in zip(actual,reference):
            torch.testing.assert_close(actual_gradient,reference_gradient)
        prepared=trainer.imagined_critic_trainer.prepare(rollout,context)
        fitted=trainer.imagined_critic_trainer.fit(prepared)
        self.assertEqual(fitted["critic_samples"],len(initial))

    def test_agent_has_no_ppo_optimizer_memory_or_real_critic(self):
        from test_world_model_fixed8_ppo import make_fixed8_agent, collect_fixed8
        legacy=make_fixed8_agent("cpu")
        env=legacy.wm_env
        config={**REFERENCE_CONFIG,"hidden":[8],"contact_hidden":[8],"amp":"none","policy_amp":"none",
            "horizon":2,"window":2,"batch_size":4,"model_microbatch_size":4,
            "imagination_batch":4,"imagination_chunk_size":2,"critic_hidden":[8],
            "critic_epochs":1,"critic_minibatches":1,"replay_envs":12,"replay_capacity":8,
            "low_replay_capacity":32,"validation_batch_size":4,"collection_warmup_high_steps":1,
            "world_model_only_iterations":0,"wm_max_batches":1,"diagnostics_interval":0}
        policy=ReferenceTeacherPolicy(1284,9,"cpu",hidden=(8,),amp="none")
        agent=ReferenceTeacherAgent(models={"policy":policy},memory=None,observation_space=1284,
            action_space=9,device="cpu",cfg={"experiment":{"write_interval":0,"checkpoint_interval":0}},
            wm_env=env,wm_config=config)
        self.assertIsNone(agent.value)
        self.assertIsNone(agent.memory)
        self.assertFalse(hasattr(agent,"optimizer"))
        actor_before=deepcopy(policy.net.state_dict())
        collect_fixed8(agent,high_steps=1)
        agent.post_interaction(0,3)
        for key,value in actor_before.items():
            torch.testing.assert_close(value,policy.net.state_dict()[key],rtol=0,atol=0)
        self.assertEqual(int(policy.obs_norm.count),12)
        collect_fixed8(agent,high_steps=2)
        agent.post_interaction(2,3)
        self.assertEqual(agent.wm_trainer.updates,1)
        self.assertEqual(agent.reference_schedule.iteration,1)
        self.assertEqual(agent.wm_ppo_updates,0)
        self.assertGreater(agent.wm_imagination.imagined_critic_trainer.updates,0)


if __name__=="__main__":
    unittest.main()
