"""CPU semantic/gradient checks for the opt-in manipulation_v2 task contract."""

import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from test_world_model_env import make_env, production_function, reference_namespace
from test_world_model_fixed8_ppo import make_fixed8_agent
from utils.wm_env import TeacherWorldModelAdapter, from_euler, quat_apply
from utils.wm_fixed8 import Fixed8TeacherWorldModelAdapter
from utils.world_model.action_contract import (horizontal_base_alignment, arm_acceleration_penalty,
                                               outer_command_deadzone)
from utils.world_model.task_objective import (ManipulationTaskObjective, collision_object_bounds,
                                             resolve_task_aux_config, raw_action_bound_cost)


def new_adapter(memory=False):
    env = make_fixed8_agent("cpu").wm_env
    env.cfg["env"]["world_model"] = dict(task_contract="manipulation_v2", task_aux_alpha=.1,
        action_coordinates="normalized_commands", critic_memory=memory,
        task_aux_config={"object_half_extents_m": {"sugar_box": [.025,.04,.08]}})
    env._low_entry_dof_vel = env._dof_vel.clone()
    return Fixed8TeacherWorldModelAdapter(env)


class ManipulationContractTests(unittest.TestCase):
    def test_base_heading_horizontal_translation_height_and_zero_distance(self):
        base = torch.zeros(5,3)
        obj = torch.tensor([[1.,0.,4.],[-1.,0.,2.],[0.,1.,8.],[.001,0.,1.],[0.,0.,0.]])
        heading = torch.tensor([[1.,0.,0.]]).expand(5,-1)
        reward = horizontal_base_alignment(base,obj,heading)
        torch.testing.assert_close(reward,torch.tensor([1.,-1.,0.,0.,0.]))
        torch.testing.assert_close(horizontal_base_alignment(base+3,obj+3,heading),reward)
        variable = obj.clone().requires_grad_()
        horizontal_base_alignment(base,variable,heading).sum().backward()
        self.assertTrue(torch.isfinite(variable.grad).all())
        self.assertGreater(variable.grad[2,0].item(),0.)

    def test_v2_real_functions_match_adapter_terms_and_independent_acc_buffer(self):
        adapter = new_adapter()
        state,context = adapter.capture()
        entry = adapter.get(state,"dof_vel")-.01
        state = adapter.replace(state,last_dof_vel=entry)
        geometry = adapter._geometry(state)
        robot,obj,ee,base,angles,heading,arm = geometry
        standin = SimpleNamespace(task_contract="manipulation_v2", num_gripper_joints=1,
            _dof_vel=adapter.get(state,"dof_vel"), _last_dof_vel=adapter.get(state,"dof_vel"),
            _low_entry_dof_vel=entry, dt=.02, _robot_root_states=robot,
            base_yaw_quat=heading, num_envs=len(state), device="cpu")
        action = adapter.get(state,"prev_action")
        terms = adapter.reward_terms(state,state,action,context,geometry=geometry)
        for name in ("base_dir","acc_penalty"):
            function = production_function(ROOT/"envs/reward_vec_task.py", "_reward_"+name,
                                            reference_namespace(),"RewardVecTask")
            actual = function(standin,obj[:,:3])[0] if name=="base_dir" else function(standin)[0]
            torch.testing.assert_close(actual,terms[name],rtol=0,atol=0)
        self.assertTrue((terms["acc_penalty"]>0).all())
        legacy = SimpleNamespace(**vars(standin)); legacy.task_contract="legacy"
        function = production_function(ROOT/"envs/reward_vec_task.py","_reward_acc_penalty",
                                       reference_namespace(),"RewardVecTask")
        self.assertEqual(function(legacy)[0].abs().max().item(),0.)

    def test_reset_restored_velocity_and_low_entry_precede_capture_all_envs(self):
        # Execute the real reset_idx body with a bank-like velocity restore.
        velocity = torch.arange(4*19,dtype=torch.float32).reshape(4,19)
        standin = SimpleNamespace(_dof_vel=velocity.clone(),_low_entry_dof_vel=torch.zeros_like(velocity))
        standin._reset_envs = lambda ids: standin._dof_vel.__setitem__(ids,-7.)
        function = production_function(ROOT/"envs/b1z1_base.py","reset_idx",reference_namespace(),"B1Z1Base")
        function(standin,torch.tensor([1,3]))
        torch.testing.assert_close(standin._low_entry_dof_vel[[1,3]],standin._dof_vel[[1,3]])
        self.assertEqual(standin._low_entry_dof_vel[[0,2]].abs().max().item(),0.)
        source=(ROOT/"envs/b1z1_base.py").read_text()
        tree=ast.parse(source)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="B1Z1Base")
        step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="step")
        text=ast.get_source_segment(source,step)
        self.assertLess(text.index("self._low_entry_dof_vel.copy_(self._dof_vel)"),text.index("wm_adapter.capture(wm_ids)"))
        refresh=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_refresh_sim_tensors")
        self.assertNotIn("_low_entry_dof_vel",ast.get_source_segment(source,refresh))

    def test_eighth_cycle_acceleration_uses_seventh_endpoint(self):
        adapter = new_adapter()
        state,context=adapter.capture()
        first=state
        delta=state.new_zeros(len(state),62)
        delta[:,43:62]=.1
        for _ in range(8):
            previous=state
            state=adapter.integrate_low(state,delta,state.new_zeros(len(state),20),context,state.new_full((len(state),1),.02))
        torch.testing.assert_close(adapter.get(state,"last_dof_vel"),adapter.get(previous,"dof_vel"))
        expected=arm_acceleration_penalty(adapter.get(state,"dof_vel")[:,12:18],adapter.get(previous,"dof_vel")[:,12:18],.02)
        actual=adapter.reward_terms(first,state,adapter.get(state,"prev_action"),context)["acc_penalty"]
        torch.testing.assert_close(actual,expected)
        self.assertGreater((adapter.get(state,"last_dof_vel")-adapter.get(first,"dof_vel")).abs().max().item(),1.)

    def test_remove_only_outer_deadzone_preserve_low_walking_and_small_gradients(self):
        adapter=new_adapter()
        state,context=adapter.capture()
        action=state.new_zeros(len(state),9); action[:,6]=.5; action[:,7]=.1; action[:,8]=.1
        action.requires_grad_()
        prepared=adapter.prepare_high(state,action)
        commands=adapter.get(prepared,"commands")
        torch.testing.assert_close(commands[:,0],torch.full((len(state),),.045))
        torch.testing.assert_close(commands[:,2],torch.full((len(state),),.06))
        _,advanced=adapter.low_observation(prepared,context)
        self.assertEqual(adapter.get(advanced,"walking").abs().max().item(),0.)
        gradient,=torch.autograd.grad(commands.sum(),action)
        torch.testing.assert_close(gradient[:,7],torch.full((len(state),),.45))
        self.assertFalse(outer_command_deadzone("manipulation_v2",True))
        self.assertTrue(outer_command_deadzone("legacy",True))

    def test_task_and_aux_post_separation_and_mean_state_gradients(self):
        adapter=new_adapter()
        previous,context=adapter.capture()
        state=previous.clone().requires_grad_()
        action=adapter.get(previous,"prev_action")
        parts=adapter.reward_components(previous,state,action,context)
        torch.testing.assert_close(parts["task"],adapter.task_reward(previous,state,action,context))
        torch.testing.assert_close(parts["total"],parts["task"]+.1*parts["aux"])
        self.assertTrue(((parts["quality"]>=0)&(parts["quality"]<=1)).all())
        gradient,=torch.autograd.grad(parts["aux"].sum(),state)
        self.assertTrue(torch.isfinite(gradient).all())
        for name in ("object","robot","dof_pos","dof_vel"):
            self.assertGreater(gradient[:,adapter.slices[name]].abs().sum().item(),0.,name)
        # PRE changes cannot move a POST-only quality. It remains separate from
        # task terms whose record/rate rewards explicitly depend on PRE.
        changed=adapter.replace(previous,object=adapter.get(previous,"object")+10.)
        torch.testing.assert_close(adapter.reward_components(changed,state,action,context)["aux"],parts["aux"])
        adapter.task_aux_alpha=0
        torch.testing.assert_close(adapter.reward(previous,state,action,context),parts["task"])

    def test_memory_candidate_exposes_only_causal_state_and_dimensions(self):
        adapter=new_adapter(memory=True)
        state,context=adapter.capture()
        variable=state.detach().requires_grad_()
        features=adapter.critic_features(variable,context)
        self.assertEqual(features.shape[-1],adapter.critic_feature_dim)
        grad,=torch.autograd.grad(features.square().sum(),variable)
        self.assertTrue(torch.isfinite(grad).all())
        for name in ("prev_action","low_history","low_actions","ekf_poses","ekf_ages","last_dof_vel"):
            changed=adapter.replace(state,**{name:adapter.get(state,name)+.01})
            self.assertGreater((adapter.critic_features(changed,context)-features).abs().sum().item(),0.,name)
        self.assertTrue(adapter.metadata()["critic_memory"])

    def test_raw_bound_cost_uses_latent_even_when_opening_is_saturated(self):
        latent=torch.zeros(2,9,requires_grad=True)
        with torch.no_grad(): latent[:,6]=30.;latent[:,7]=2.
        action=torch.cat((latent[:,:6],latent[:,6:7].sigmoid(),latent[:,7:]),-1)
        cost=raw_action_bound_cost(action,latent,.45,coordinates="normalized_commands",latent_bound=3.)
        torch.testing.assert_close(cost,torch.full((2,),730.))
        gradient,=torch.autograd.grad(cost.sum(),latent)
        torch.testing.assert_close(gradient[:,6],torch.full((2,),54.))
        with self.assertRaisesRegex(ValueError,"original latent"):
            raw_action_bound_cost(action,None,.45,coordinates="normalized_commands")


class TaskObjectiveGeometryTests(unittest.TestCase):
    def objective_inputs(self):
        env=make_env("cpu",n=1)
        objective=ManipulationTaskObjective(env,{"object_half_extents_m":{"sugar_box":[.02,.02,.02]},
            "object_centers_m":{"sugar_box":[0.,0.,0.]}})
        identity=torch.eye(3)[None]
        robot=torch.zeros(1,13);robot[:,6]=1.;robot[:,2]=.55
        ee=robot.clone();ee[:,:3]=torch.tensor([[0.,0.,.5]])
        obj=ee.clone();obj[:,:3]+=torch.tensor([[-.009,0.,.01]])
        inputs=dict(robot=robot,obj=obj,ee=ee,object_rotation=identity,ee_rotation=identity,
            grasp_positions=ee[:,None,:3].clone(),grasp_rotations=identity[:,None],
            gripper_angle=torch.tensor([-.5]),object_ids=torch.zeros(1,dtype=torch.long),
            table_height=torch.tensor([.25]),rest_height=torch.tensor([.05]),lift_threshold=.2,high_dt=.16)
        return objective,inputs

    def test_slip_rejects_false_following_and_lift_is_bounded(self):
        objective,inputs=self.objective_inputs()
        stable=objective(**inputs)
        moved=dict(inputs,obj=inputs["obj"].clone());moved["obj"][:,7]=1.
        slip=objective(**moved)
        self.assertLess(slip["grasp"].item(),stable["grasp"].item())
        self.assertGreater(slip["slip_cost"].item(),stable["slip_cost"].item())
        high=dict(inputs,table_height=torch.tensor([-.5]))
        high=objective(**high)
        self.assertEqual(high["lift"].item(),1.)
        self.assertGreater(high["unsafe_cost"].item(),stable["unsafe_cost"].item())
        self.assertLess(high["aux"].item(),stable["aux"].item())

    def test_closed_gripper_nearby_is_not_geometry_evidence(self):
        objective,inputs=self.objective_inputs()
        good=objective(**inputs)
        bad=dict(inputs,gripper_angle=torch.zeros(1))
        closed=objective(**bad)
        self.assertLess(closed["grasp_geometry"].item(),good["grasp_geometry"].item())
        far=dict(inputs,ee=inputs["ee"].clone());far["ee"][:,0]+=1.
        self.assertLess(objective(**far)["slip_cost"].item(),1e-5)

    def test_collision_bounds_and_configuration_are_explicit(self):
        center,half=collision_object_bounds(ROOT/"data/asset/obj_set/sugar_box/model.urdf")
        self.assertTrue(all(math.isfinite(v) for v in center+half))
        self.assertTrue(all(.001<v<.5 for v in half))
        for config in ({"typo":1},{"slip_linear_scale_m_s":0},{"quality_weights":[1,1,1,1,1]}):
            with self.assertRaises(ValueError):resolve_task_aux_config(config)


if __name__=="__main__":
    unittest.main()
