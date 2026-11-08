"""CPU contracts for finite real-rollout value and action-direction reports."""
import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tests")]
from utils.world_model.task_diagnostics import (finite_episode_returns,diagnostic_policy_mean,
    diagnostic_noise_seed,value_diagnostic_report,paired_direction_report)
from utils.world_model.policy_inputs import HighLevelObservationNoise,FixedScale
from test_world_model_ppo import TinyPolicy


class TaskDiagnosticTests(unittest.TestCase):
    def test_terminal_reward_kept_timeout_zero_bootstrap_and_censor(self):
        rewards=torch.tensor([[1.,1.,1.,9.],[2.,2.,2.,9.],[100.,100.,3.,9.]])
        terminated=torch.tensor([[False,False,False,True],[True,False,False,False],[False,False,False,False]])
        timeouts=torch.tensor([[False]*4,[False,True,False,False],[False]*4])
        returns,completed,length=finite_episode_returns(rewards,terminated,timeouts,.5,torch.tensor([True,True,True,False]))
        torch.testing.assert_close(returns,torch.tensor([2.,2.,2.75,0.]))
        self.assertEqual(completed.tolist(),[True,True,False,False])
        self.assertEqual(length.tolist(),[2,2,3,0])
        report=value_diagnostic_report(torch.tensor([2.,3.,999.,999.]),returns,completed,
                                       torch.tensor([True,True,True,False]))
        self.assertEqual(report["complete"],2)
        self.assertEqual(report["unfinished"],1)
        self.assertEqual(report["error"]["mean"],.5)

    def test_nonfinite_live_reward_rejected_absorbed_padding_ignored(self):
        reward=torch.tensor([[1.,2.],[float("nan"),3.]])
        done=torch.tensor([[True,False],[False,True]])
        result,completed,_=finite_episode_returns(reward,done,torch.zeros_like(done))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(completed.all())
        with self.assertRaises(FloatingPointError):
            finite_episode_returns(reward,torch.zeros_like(done),torch.zeros_like(done))

    def test_mean_uses_same_sensor_noise_and_keeps_rng_scaler_std_unchanged(self):
        policy=TinyPolicy(6,"cpu")
        noise=HighLevelObservationNoise(6,"cpu",noise_half_width=.1,noise_scale=1.).train()
        scaler=FixedScale(6,"cpu",scale=2.)
        observation=torch.zeros(4,6)
        rng=torch.random.get_rng_state().clone()
        before={key:value.clone() for key,value in policy.state_dict().items()}
        first,action=diagnostic_policy_mean(policy,scaler,observation,noise=noise,seed=5)
        repeat,repeated_action=diagnostic_policy_mean(policy,scaler,observation,noise=noise,seed=5)
        different,_=diagnostic_policy_mean(policy,scaler,observation,noise=noise,seed=6)
        torch.testing.assert_close(first,repeat,rtol=0,atol=0)
        torch.testing.assert_close(action,repeated_action,rtol=0,atol=0)
        self.assertGreater((first-different).abs().sum().item(),0.)
        torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0,atol=0)
        for key,value in policy.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        self.assertTrue(((action[:,6]>0)&(action[:,6]<1)).all())

    def test_real_direction_and_repeatability_disagreement_is_visible(self):
        base=torch.zeros(3)
        report=paired_direction_report(base,torch.tensor([1.,-1.,.01]),torch.tensor([-1.,1.,-.01]),
            torch.tensor([0.,0.,1.]),torch.ones(3),-torch.ones(3),torch.ones(3,dtype=torch.bool))
        self.assertEqual(report["informative_pairs"],2)
        self.assertEqual(report["direction_agreement"],.5)
        self.assertEqual(report["model_improves_real_worsens"],1)

    def test_cli_import_is_simulator_free_and_exposes_all_nine_coordinates(self):
        spec=importlib.util.spec_from_file_location("diagnose_manipulation_teacher",ROOT/"diagnose_manipulation_teacher.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for coordinate in range(9):
            args=module.parser().parse_args(["--checkpoint","x","--output","y","--direction","coordinate",
                                            "--coordinate",str(coordinate)])
            self.assertEqual(args.coordinate,coordinate)

    def test_first_mean_bptt_probe_runs_four_by_eight_and_keeps_std_unowned(self):
        from test_world_model_manipulation_task import new_adapter
        from test_world_model_ppo import TinyValue
        from utils.world_model.hierarchical import HierarchicalDynamics
        from utils.world_model.imagination import ImaginationConfig,ImaginationTrainer
        import diagnose_manipulation_teacher as diagnostic
        adapter=new_adapter()
        state,context=adapter.capture()
        model=HierarchicalDynamics(adapter,hidden=(8,8),contact_hidden=(8,),device="cpu")
        with torch.no_grad():
            list(model.dynamics.parameters())[-2].normal_(0,1e-5)
        model.requires_grad_(False)
        policy=TinyPolicy(1284,"cpu")
        trainer=ImaginationTrainer(model,adapter,policy,TinyValue(1284,"cpu"),
            ImaginationConfig(horizon=4,dt=.16,value_source="imagined",critic_hidden=(8,),kl_rollback=False))
        trainer.imagined_critic.requires_grad_(False)
        first=torch.zeros(len(state),9,requires_grad=True)
        std=policy.log_std.detach().clone()
        with patch.object(adapter,"low_control",wraps=adapter.low_control) as low:
            predicted=diagnostic._imagine_first_mean(SimpleNamespace(seed=7,prefix_steps=0),
                                                      trainer,state,context,first)
            self.assertEqual(low.call_count,32)
        gradient,=torch.autograd.grad(predicted.sum(),first)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(),0.)
        torch.testing.assert_close(policy.log_std,std,rtol=0,atol=0)
        self.assertIsNone(policy.log_std.grad)


if __name__=="__main__":unittest.main()
