"""Task adapter parity and differentiability on CPU and CUDA.

The expected observation executes the production observation and grasp-template
functions, and uses the original no-grad EKF. No simulator is mocked inside
the adapter itself; a tensor-only environment fixture supplies captured state.
"""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from modules.pose_prediction import RollingSE3Predictor
from PredictPiontTransform.predictpoint_mul import PredictPoint
from utils.wm_env import (TeacherWorldModelAdapter, causal_ekf, from_euler,
                          quat_apply, quat_conjugate, quat_inverse_apply,
                          quat_mul, euler)
from utils.pick_success import advance_pick_hold, lifted_grasp_mask


ROOT = Path(__file__).resolve().parents[1]


def production_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    if class_name:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    code = ast.Module(body=[node], type_ignores=[])
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)
    return scope[name]


def reference_namespace():
    ns = {"torch": torch, "Tensor": torch.Tensor, "F": torch.nn.functional,
          "quat_apply": quat_apply, "quat_rotate_inverse": quat_inverse_apply,
          "quat_rotate_inverse_batched": quat_inverse_apply, "quat_conjugate": quat_conjugate,
          "quat_mul": quat_mul, "euler_from_quat": lambda q: euler(q).unbind(-1),
          "quat_from_euler_xyz": lambda r,p,y: from_euler(torch.stack((r,p,y), -1)),
          "COMMAND_SMOOTH_THRESHOLD": .2}
    ns["reindex_all"] = production_function(ROOT/"envs/b1z1_base.py", "reindex_all", ns)
    ns["object_pose_in_teacher_frame"] = production_function(ROOT/"utils/object_pose.py", "object_pose_in_teacher_frame", ns)
    ns["quat_to_euler_zyx"] = production_function(ROOT.parent/"third_party/isaacgym/python/isaacgym/torch_utils.py", "quat_to_euler_zyx", ns)
    return ns


def make_env(device="cpu", n=6, velocity_obs=False, count=5):
    torch.manual_seed(7)
    env = SimpleNamespace(device=device, num_envs=n, num_actions=9, num_gripper_dof=1,
        use_roboinfo=True, table_level="Level02", commands_curriculum=True, stop_pick=False,
        max_episode_length=100, hold_steps=3, lifted_success_threshold=.2,
        base_object_distace_threshold=.6, clip_actions=1., clip_obs=float("inf"), dt=.02,
        num_velocity_obs=5 if velocity_obs else 0, num_future_pose_obs=13,
        pose_prediction_horizons=(.6,.9), obj_list=["sugar_box"], rand_cmd_scale=False,
        global_step_counter=100, train_reward_strict=50, control_freq_low_init=8, rand_control=True)
    env.cfg = {"env": {"smallValueSetZero": True, "posePrediction": {"qScale":1., "rScale":.01}},
               "reward": {"base_height_target":.55, "only_positive_rewards": False}}
    env.reward_scales = {"approaching":.5, "lifting":.8, "pick_up":50., "acc_penalty":-.001,
                        "command_reward":.05, "action_rate":-.001, "ee_orn":.01, "base_dir":.05,
                        "base_approaching":.01, "grasp_base_height":.5, "limit_yaw_rotation_penalty":.4}
    zero = lambda *shape: torch.zeros(*shape, device=device)
    env._robot_root_states = zero(n,13)
    env._robot_root_states[:, 0] = torch.arange(n, device=device)*8
    env._robot_root_states[:, 2] = .55
    env._robot_root_states[:, 3:7] = from_euler(torch.randn(n,3,device=device)*.05)
    env._robot_root_states[:, 7:] = torch.randn(n,6,device=device)*.03
    env._initial_robot_root_states = env._robot_root_states.clone()
    env.initial_yaw_z = euler(env._robot_root_states[:, 3:7])[:, 2]
    env._cube_root_states = env._robot_root_states.clone()
    env._cube_root_states[:, :3] += torch.tensor([.6, 0., .2], device=device)
    env._cube_root_states[:, 3:7] = from_euler(torch.randn(n,3,device=device)*.15)
    env._table_root_states = env._cube_root_states.clone()
    env._table_root_states[:, 2] = .5
    env._table_root_states[:, 3:7] = from_euler(zero(n,3))
    env._dof_pos, env._dof_vel = torch.randn(n,19,device=device)*.1, torch.randn(n,19,device=device)*.05
    env._last_dof_vel = env._dof_vel.clone()
    env.low_obs_history_buf = torch.randn(n,10,76,device=device)*.05
    env.last_low_actions = torch.randn(n,18,device=device)*.1
    env.gait_indices, env.clock_inputs = zero(n), zero(n,4)
    env.gait_wait_timer, env.is_walking = zero(n), zero(n)
    env.foot_contacts_from_sensor = zero(n,4)
    env._rigid_body_pos = (env._cube_root_states[:, :3]-torch.tensor([.04,0.,0.],device=device))[:, None]
    env._rigid_body_rot = env._robot_root_states[:, None, 3:7].clone()
    env._rigid_body_vel, env._rigid_body_ang_vel = zero(n,1,3), zero(n,1,3)
    env.gripper_idx = 0
    arm = env._robot_root_states[:, :3]+quat_apply(env._robot_root_states[:, 3:7],torch.tensor([.3,0.,.09],device=device).expand(n,-1))
    env.curr_ee_goal_cart = quat_inverse_apply(env._robot_root_states[:, 3:7], env._rigid_body_pos[:,0]-arm)
    env.curr_ee_goal_orn_rpy = zero(n,3)
    env.commands = zero(n,3)
    env.action_history_buf = zero(n,3,9)
    env.action_history_buf[:, -1, 6] = -1.
    env.closest_dist, env.highest_object = torch.full((n,),.06,device=device), zero(n)
    env.pick_counter, env.best_pick_counter = zero(n), zero(n)
    env.pick_success_recorded = torch.zeros(n, device=device, dtype=torch.bool)
    env.progress_buf = torch.ones(n,device=device,dtype=torch.long)*8
    env.table_heights, env.init_height = torch.full((n,),.6,device=device), torch.full((n,),.05,device=device)
    env.motor_strength = torch.ones(n,18,device=device)
    env.vel_line_forward_y_level02 = torch.full((n,),.2,device=device)
    env.vel_angular_yaw = torch.full((n,),.1,device=device)
    env.lin_circle_select = torch.arange(n,device=device)%2==0
    env.feature_obs = torch.randn(n,1024,device=device)
    env.wm_body_context = [(20.,.02,-.01,0.)]*n
    env.wm_shape_context = [(400.,40.,3.,2.5,1.,2.,1.5)]*n
    env.Rx_rot = torch.tensor([[1,0,0],[0,0,-1],[0,1,0]],device=device,dtype=torch.float32)
    env.Rz_rot = torch.tensor([[0,-1,0],[1,0,0],[0,0,1]],device=device,dtype=torch.float32)
    point = PredictPoint.__new__(PredictPoint)
    point.Rx_rot, point.Rz_rot = env.Rx_rot, env.Rz_rot
    transforms = torch.eye(4,device=device).expand(n,30,-1,-1).clone()
    transforms[:,:,:3,:3] = point._rpy_to_matrix(torch.randn(n*30,3,device=device)*.2).reshape(n,30,3,3)
    transforms[:,:,:3,3] = torch.randn(n,30,3,device=device)*.03
    env.T_grasp_in_obj = transforms
    point.T_grasp_in_obj = transforms
    env.point = point
    env.object_pose_predictor = RollingSE3Predictor(n,device,history=5,horizons=(.6,.9))
    for i in range(count):
        pose = env._cube_root_states[:,:7].clone()
        pose[:,0] -= (count-i-1)*.003
        env.object_pose_predictor.push(pose, .15*i)
    env.object_pose_predictor.predict()
    return env


def production_observation(env):
    ns = reference_namespace()
    compute = production_function(ROOT/"envs/b1z1_pickmulti.py", "compute_robot_observations", ns)
    n, device = env.num_envs, env.device
    angles = euler(env._robot_root_states[:,3:7])
    heading = from_euler(torch.cat((torch.zeros_like(angles[:,:2]), angles[:,2:]), -1))
    arm = env._robot_root_states[:,:3]+quat_apply(env._robot_root_states[:,3:7],torch.tensor([.3,0.,.09],device=device).expand(n,-1))
    obs = compute(env._robot_root_states,env._table_root_states,env._cube_root_states,
        env._rigid_body_pos,env._rigid_body_rot,env._rigid_body_vel,env._rigid_body_ang_vel,
        env._dof_pos,env._dof_vel,heading,arm,env.commands,env.gripper_idx,
        torch.ones(n,3,device=device),env.curr_ee_goal_cart,env.curr_ee_goal_orn_rpy,arm,
        env._robot_root_states[:,3:7],True,False)
    if not env.num_velocity_obs:
        obs = torch.cat((obs[:,:-11],obs[:,-6:]), -1)
    future, valid = env.object_pose_predictor.future_poses, env.object_pose_predictor.valid
    future_obs = ns["object_pose_in_teacher_frame"](future[:,:,:3].reshape(-1,3),future[:,:,3:].reshape(-1,4),
        arm[:,None].expand(-1,2,-1).reshape(-1,3),heading[:,None].expand(-1,2,-1).reshape(-1,4)).reshape(n,2,6)
    future_obs = torch.where(valid[:,None,None],future_obs,obs[:,:6,None].transpose(1,2))
    position, rpy = env.point.get_grasp_world(obs[:,-6:],torch.arange(n,device=device))
    quaternion = from_euler(rpy)
    base = env._robot_root_states[:,None,3:7].expand(-1,30,-1)
    rotation = ns["quat_to_euler_zyx"](quat_mul(quat_conjugate(base),quaternion))
    position = quat_inverse_apply(base,position-arm[:,None])
    position[:,:,2] = position[:,:,2].clamp(-.6,.6)
    return torch.cat((env.feature_obs,obs[:,:-6],future_obs.flatten(1),valid[:,None].float(),position.flatten(1),rotation.flatten(1),env.action_history_buf[:,-1]), -1)


class AdapterTests(unittest.TestCase):
    def devices(self):
        return ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])

    def test_observation_matches_production_layout_and_causal_ekf(self):
        for device in self.devices():
            for velocity in (False,True):
                for count in (1,5):
                    with self.subTest(device=device,velocity=velocity,count=count):
                        env = make_env(device,velocity_obs=velocity,count=count)
                        adapter = TeacherWorldModelAdapter(env)
                        state, context = adapter.capture()
                        actual = adapter.observation(state,context)
                        expected = production_observation(env)
                        torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-5)
                        self.assertEqual(actual.shape[-1],1289 if velocity else 1284)
                        self.assertEqual(adapter.normalize_context(context).shape[-1],adapter.context_feature_dim)

    def test_ekf_matches_original_and_has_finite_history_gradients(self):
        for device in self.devices():
            env = make_env(device,n=3)
            predictor = env.object_pose_predictor
            poses = predictor.poses.clone().requires_grad_()
            ages = (predictor.times-predictor.times[:,-1:]).float()
            future,valid = causal_ekf(poses,ages,predictor.count,(.6,.9))
            torch.testing.assert_close(future,predictor.future_poses,atol=2e-5,rtol=2e-5)
            self.assertTrue(valid.all().item())
            future[:,:,:3].square().sum().backward()
            self.assertTrue(torch.isfinite(poses.grad).all().item())
            self.assertGreater(poses.grad[:,:-1].abs().sum().item(),0.)

    def test_multistep_integration_observation_and_reward_backward(self):
        for device in self.devices():
            env = make_env(device,n=3)
            adapter = TeacherWorldModelAdapter(env)
            start,context = adapter.capture()
            state = start.clone().requires_grad_()
            actions = torch.full((3,9),.01,device=device,requires_grad=True)
            delta = torch.zeros(3,adapter.delta_dim,device=device,requires_grad=True)
            first = adapter.integrate(state,delta,actions,context,torch.full((3,1),.14,device=device))
            second = adapter.integrate(first,delta,actions,context,torch.full((3,1),.16,device=device))
            loss = adapter.observation(second,context)[:,1024:].square().mean()+adapter.reward(first,second,actions,context).mean()
            loss.backward()
            for tensor in (state,actions,delta):
                self.assertTrue(torch.isfinite(tensor.grad).all().item())
                self.assertGreater(tensor.grad.abs().sum().item(),0.)
            torch.testing.assert_close(adapter.get(second,"ekf_ages")[:,-2:],torch.tensor([[-.16,0.]],device=device).expand(3,-1))
            torch.testing.assert_close(adapter.get(second,"prev_action"),actions.detach())

    def test_reward_matches_existing_production_functions(self):
        for device in self.devices():
            env = make_env(device,n=4)
            adapter = TeacherWorldModelAdapter(env)
            previous,context = adapter.capture()
            state = previous.clone()
            state[:,adapter.slices["object"].start+2] += torch.tensor([.02,.12,.12,.02],device=device)
            state[:,adapter.slices["ee"].start+2] += torch.tensor([.02,.12,.12,.02],device=device)
            state[:,adapter.slices["hold"]] = 0
            previous[2,adapter.slices["hold"]] = 2
            previous[2,adapter.slices["best_hold"]] = 2
            action = adapter.get(previous,"prev_action").clone()
            outcome = adapter.task_outcome(previous,state,action,context)
            robot,obj,ee,base,angles,heading,arm = adapter._geometry(state)
            standin = SimpleNamespace(num_envs=len(state),device=device,
                closest_dist=adapter.get(previous,"closest")[:,0].clone(), curr_dist=outcome["distance"],
                highest_object=adapter.get(previous,"highest")[:,0].clone(),curr_height=outcome["height"],
                lifted_object=outcome["lifted"],pick_success=outcome["success"], pick_hold_progress=outcome["hold_progress"],
                _dof_vel=adapter.get(state,"dof_vel"),_last_dof_vel=adapter.get(state,"last_dof_vel"),num_gripper_joints=1,
                dt=.02,commands=adapter.get(state,"commands"),actions=action,last_actions=adapter.get(previous,"prev_action"),
                arm_base=arm,ee_pos=ee[:,:3],ee_orn=ee[:,3:7],_robot_root_states=robot,_cube_root_states=obj,
                base_yaw_quat=heading,base_object_distace_threshold=.6,curr_ee_goal_cart=adapter.get(state,"goal_pos"),
                cfg=env.cfg,global_step_counter=100,train_reward_strict=50)
            ns = reference_namespace()
            expected = torch.zeros(len(state),device=device)
            terms = adapter.reward_terms(previous,state,action,context)
            for name,scale in env.reward_scales.items():
                if name=="grasp_base_height":
                    fn = production_function(ROOT/"envs/reward_vec_task.py","_reward_base_height",ns,"RewardVecTask")
                    value = fn(standin)[0]*outcome["hold_progress"]
                elif name=="limit_yaw_rotation_penalty":
                    standin._yaw_deviation=lambda: (angles[:,2]-context[:,1]).abs()
                    fn = production_function(ROOT/"envs/b1z1_pickmulti.py","_reward_"+name,dict(ns,np=SimpleNamespace(deg2rad=lambda v:v*math.pi/180)),"B1Z1PickMulti")
                    value = fn(standin)[0]
                else:
                    fn = production_function(ROOT/"envs/reward_vec_task.py","_reward_"+name,ns,"RewardVecTask")
                    value = fn(standin,obj[:,:3])[0] if name in ("command_reward","ee_orn","base_dir","base_approaching") else fn(standin)[0]
                    if name=="approaching":
                        value=value*~outcome["lifted"]
                torch.testing.assert_close(terms[name],value,atol=1e-6,rtol=1e-5)
                expected += value*scale
            torch.testing.assert_close(adapter.reward(previous,state,action,context),expected,atol=1e-6,rtol=1e-5)
            self.assertTrue(outcome["success"][2].item())

    def test_failure_precedes_success_and_success_precedes_timeout(self):
        for device in self.devices():
            env = make_env(device,n=3)
            adapter = TeacherWorldModelAdapter(env)
            previous,context = adapter.capture()
            previous[:,adapter.slices["hold"]] = 2
            previous[:,adapter.slices["best_hold"]] = 2
            state = previous.clone()
            state[:,adapter.slices["progress"]] = adapter.max_steps
            state[:,adapter.slices["object"].start+2] = .9
            state[:,adapter.slices["ee"].start+2] = .9
            state[:,adapter.slices["goal_pos"].start+2] = .9-.55-.09
            # One hard robot fall; one valid completion; one open-gripper time limit.
            state[0,adapter.slices["robot"].start+2] = .05
            action = adapter.get(previous,"prev_action").clone()
            action[2,6] = 1.
            result = adapter.task_outcome(previous,state,action,context)
            self.assertEqual(result["failed"].tolist(),[True,False,False])
            self.assertEqual(result["success"].tolist(),[False,True,False])
            self.assertEqual(result["timeout"].tolist(),[False,False,True])
            self.assertEqual(adapter.reward_terms(previous,state,action,context)["pick_up"].tolist(),[0.,1.,0.])

    def test_capture_accepts_production_scalar_initial_heading(self):
        for device in self.devices():
            env = make_env(device,n=3)
            env.initial_yaw_z = torch.tensor([0.],device=device)
            adapter = TeacherWorldModelAdapter(env)
            _,context = adapter.capture(torch.tensor([2,0],device=device))
            self.assertEqual(context[:,0].tolist(),[2.,0.])
            self.assertEqual(context[:,1].tolist(),[0.,0.])

    def test_metadata_stable_across_device_environment_count_and_random_samples(self):
        reference = None
        for device in self.devices():
            for count in (3,7):
                env = make_env(device,n=count)
                env.cfg["env"].update(numEnvs=count,world_model={"enabled": count==3},logdir=str(count))
                env.wm_body_context = [(10.+count,.01*count,0.,0.)]*count
                adapter = TeacherWorldModelAdapter(env)
                metadata = adapter.metadata()
                encoded = json.dumps(metadata,sort_keys=True,allow_nan=False)
                self.assertEqual(metadata["clip_observations"],"Infinity")
                if reference is None:
                    reference = encoded
                else:
                    self.assertEqual(reference,encoded)
                adapter.hold_steps += 1
                self.assertNotEqual(encoded,json.dumps(adapter.metadata(),sort_keys=True,allow_nan=False))

    def test_physical_action_encoding_has_no_gradient_through_saturated_commands(self):
        for device in self.devices():
            env = make_env(device,n=3)
            adapter = TeacherWorldModelAdapter(env)
            state,_ = adapter.capture()
            action = torch.full((3,9),.9,device=device,requires_grad=True)
            physical = adapter.dynamics_action(state,action)
            physical.sum().backward()
            self.assertTrue(torch.isfinite(action.grad).all().item())
            self.assertEqual(action.grad.abs().sum().item(),0.)
            torch.testing.assert_close(physical[:,:3],torch.full((3,3),.02,device=device),atol=1e-7,rtol=0.)
            self.assertEqual(physical[:,6].tolist(),[1.,1.,1.])

    def test_dynamics_inputs_exclude_nonphysical_high_level_history(self):
        for device in self.devices():
            adapter = TeacherWorldModelAdapter(make_env(device,n=3))
            state,_ = adapter.capture()
            state.requires_grad_()
            adapter.features(state).sum().backward()
            for name in ("prev_action","ekf_poses","ekf_ages","ekf_count","closest","highest","hold","best_hold","completed"):
                self.assertEqual(state.grad[:,adapter.slices[name]].abs().sum().item(),0.)
            for name in ("low_history","low_actions","foot_contacts","goal_pos","goal_rpy","commands","progress"):
                self.assertGreater(state.grad[:,adapter.slices[name]].abs().sum().item(),0.)

    def test_relative_loss_zero_at_parity_invariant_to_quaternion_sign_and_finite_gradient(self):
        for device in self.devices():
            adapter = TeacherWorldModelAdapter(make_env(device,n=3))
            state,_ = adapter.capture()
            self.assertEqual(adapter.auxiliary_loss(state,state).abs().sum().item(),0.)
            sign_changed=state.clone()
            for name in ("object","ee"):
                start=adapter.slices[name].start
                sign_changed[:,start+3:start+7] *= -1
                self.assertEqual(adapter.auxiliary_loss(sign_changed,state).abs().sum().item(),0.)
            prediction=state.clone()
            prediction[:,adapter.slices["object"].start] += .025
            prediction[:,adapter.slices["ee"].start+7] += .1
            prediction.requires_grad_()
            loss=adapter.auxiliary_loss(prediction,state).mean()
            self.assertGreater(float(loss),0.)
            loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all().item())
            self.assertGreater(float(prediction.grad[:,adapter.slices["object"]].abs().sum()),0.)
            self.assertGreater(float(prediction.grad[:,adapter.slices["ee"]].abs().sum()),0.)


if __name__ == "__main__":
    unittest.main()
