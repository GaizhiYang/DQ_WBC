#!/usr/bin/env python3
"""Independent real mean-policy value and first-action BPTT direction diagnostics.

Example (dqwbc environment)::

  python diagnose_manipulation_teacher.py --checkpoint candidate.pt \
      --output diagnostics.json --num_envs 64 --direction bptt --max_steps 5000

Each worker owns a fresh Isaac Gym process. A recorded prefix is replayed;
the training simulator is never rewound. Complete real finite-task returns
use the same imagined reward (including auxiliary and optional raw costs),
while native task returns are reported separately. H-step model-plus-tail
predictions remain model estimates. No calibrated acceptance threshold or
training improvement is inferred by this tool.
"""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--num_envs",type=int,default=64)
    p.add_argument("--prefix_steps",type=int,default=4)
    p.add_argument("--max_steps",type=int,default=5000,help="Diagnostic cutoff; unfinished rows are never value truth")
    p.add_argument("--seed",type=int,default=43)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--direction",choices=("bptt","coordinate","random"),default="bptt")
    p.add_argument("--coordinate",type=int,choices=range(9),default=6)
    p.add_argument("--radius",type=float,default=.05,help="Perturbation of raw Actor output before the gripper sigmoid")
    p.add_argument("--family_sampling",choices=("balanced","original"),default="original")
    p.add_argument("--worker_timeout",type=float,default=3600.)
    p.add_argument("--workdir",default=None)
    p.add_argument("--worker",choices=("plan","base","plus","minus","repeat"),default=None,help=argparse.SUPPRESS)
    p.add_argument("--bundle_sha256",default=None,help=argparse.SUPPRESS)
    return p


def _noise_seed(cli,step):
    return cli.seed+100003*(step+1)


def _teacher_gamma(agent):
    settings = getattr(agent, "wm_config", {})
    return float(settings["gamma"] if settings.get("recipe") == "reference" else agent._discount_factor)


def _real_mean(cli,session,step):
    from utils.world_model.task_diagnostics import diagnostic_policy_mean
    env,agent=session.env,session.trainer.agents
    obs=env.obs_buf.clamp(-env.clip_obs,env.clip_obs)
    return diagnostic_policy_mean(agent.policy,getattr(agent,"_state_preprocessor",None),obs,
        noise=getattr(env,"world_model_observation_noise",None),seed=_noise_seed(cli,step))


def _reset_ended(session,done):
    """Recycle only finished isolated worlds; their later data is excluded."""
    ids=done.nonzero(as_tuple=False).flatten()
    if len(ids):
        session.env.reset_idx(ids)


def _make_imagination(session,bundle,device):
    from probe_teacher_world_model import _load_probe_model
    from utils.world_model.imagination import ImaginationConfig,ImaginationTrainer
    from utils.world_model.task_diagnostics import diagnostic_noise_seed
    agent=session.trainer.agents
    cfg=bundle["config"]
    reference=cfg.get("recipe")=="reference"
    if not bundle.get("imagined_critic"):
        raise ValueError("Candidate lacks its matching imagined Critic; re-export a new candidate bundle")
    if cfg.get("value_source","imagined")!="imagined" or bundle.get("imagined_bootstrap_timeouts",cfg.get("bootstrap_timeouts",False)):
        raise ValueError("This diagnostic requires the finite-task imagined Critic, without timeout bootstrap")
    gamma=float(bundle.get("imagined_gamma",_teacher_gamma(agent)))
    if not math.isclose(gamma,_teacher_gamma(agent),rel_tol=0,abs_tol=1e-12):
        raise ValueError("Reconstructed real policy discount differs from saved imagined Critic discount")
    with diagnostic_noise_seed(991,device):
        model=_load_probe_model(session.adapter,bundle,device)
        trainer=ImaginationTrainer(model,session.adapter,agent.policy,getattr(agent,"value",None),
            ImaginationConfig(horizon=int(cfg.get("horizon",4)),gamma=gamma,
                dt=.16,value_source="imagined",critic_hidden=tuple(cfg.get("critic_hidden",[256,128])),
                use_tail_value=bundle.get("imagined_use_tail_value",cfg.get("use_tail_value",True)),kl_rollback=False,
                reference_mode=reference,
                objective_reduction=cfg.get("actor_objective_reduction","sum" if reference else "mean"),
                critic_running_norm=cfg.get("critic_running_normalization",reference),
                critic_activation=cfg.get("critic_activation","elu" if reference else "silu"),
                critic_amp=cfg.get("policy_amp","bf16") if reference else "none"),
            state_preprocessor=getattr(agent,"_state_preprocessor",None),
            value_preprocessor=getattr(agent,"_value_preprocessor",None))
        trainer.imagined_critic.load_state_dict(bundle["imagined_critic"])
    model.requires_grad_(False)
    trainer.imagined_critic.requires_grad_(False)
    trainer.imagined_critic.eval()
    return trainer


def _imagine_first_mean(cli,trainer,state,context,first_mean):
    """Same BPTT rollout, replacing only the first raw Actor output for probing."""
    import torch
    from utils.world_model.task_diagnostics import diagnostic_noise_seed
    adapter=trainer.adapter
    policy_fn=trainer._policy
    observation_fn=adapter.policy_observation
    calls=[0]
    noise_calls=[0]
    def policy(observation):
        if calls[0]==0:
            calls[0]+=1
            return first_mean,torch.zeros_like(first_mean)
        calls[0]+=1
        return policy_fn(observation)
    def observation(current,condition):
        step=cli.prefix_steps+noise_calls[0]
        noise_calls[0]+=1
        with diagnostic_noise_seed(_noise_seed(cli,step),current.device):
            return observation_fn(current,condition)
    trainer._policy=policy
    adapter.policy_observation=observation
    try:
        rollout=trainer.imagine(state,context,dt=.16)
    finally:
        trainer._policy=policy_fn
        adapter.policy_observation=observation_fn
    discounts=trainer.cfg.gamma**torch.arange(trainer.cfg.horizon,device=state.device,dtype=state.dtype)
    predicted=(rollout.rewards*rollout.survival[:-1]*discounts[:,None]).sum(0)
    predicted=predicted+trainer.cfg.gamma**trainer.cfg.horizon*rollout.survival[-1]*rollout.tail_values
    return predicted


def _plan(cli,session,bundle,torch):
    prefix=[]
    alive=torch.ones(cli.num_envs,device=cli.device,dtype=torch.bool)
    for step in range(cli.prefix_steps):
        with torch.no_grad():
            _,action=_real_mean(cli,session,step)
            prefix.append(action.detach().clone())
            result=session.step(action)
            alive &= ~result.terminated
            _reset_ended(session,result.terminated)
    state,context=session.capture()
    with torch.no_grad():
        mean,_=_real_mean(cli,session,cli.prefix_steps)
    trainer=_make_imagination(session,bundle,cli.device)
    first=mean.detach().requires_grad_(True)
    predicted=_imagine_first_mean(cli,trainer,state,context,first)
    gradient,=torch.autograd.grad(predicted.sum(),first)
    if not bool(torch.isfinite(gradient).all() & torch.isfinite(predicted).all()):
        raise FloatingPointError("Nonfinite BPTT first-action direction; diagnostic cannot continue")
    if cli.direction=="bptt":
        direction=gradient/torch.linalg.vector_norm(gradient,dim=-1,keepdim=True).clamp_min(1e-12)
    elif cli.direction=="coordinate":
        direction=torch.zeros_like(mean);direction[:,cli.coordinate]=1.
    else:
        generator=torch.Generator(device=cli.device).manual_seed(cli.seed+2911)
        direction=torch.randn(mean.shape,device=cli.device,dtype=mean.dtype,generator=generator)
        direction=direction/torch.linalg.vector_norm(direction,dim=-1,keepdim=True).clamp_min(1e-12)
    with torch.no_grad():
        value=trainer.imagined_critic(session.adapter.critic_features(state,context))
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite saved imagined Critic value")
    predictions={}
    # imagine explicitly enables gradients; discard each graph immediately.
    for name,multiplier in (("base",0),("plus",1),("minus",-1)):
        predictions[name]=_imagine_first_mean(cli,trainer,state,context,
            mean.detach()+multiplier*cli.radius*direction.detach()).detach()
    record={"prefix":torch.stack(prefix) if prefix else state.new_empty((0,cli.num_envs,9)),
            "mean":mean,"direction":direction,"gradient":gradient,"state":state,"context":context,
            "initial_alive":alive,"initial_value":value,"families":session.probe_families,
            **{"model_"+name:value for name,value in predictions.items()}}
    torch.save({key:value.detach().cpu() for key,value in record.items()},Path(cli.workdir)/"plan.pt")


def _real_branch(cli,session,bundle,torch):
    from utils.world_model.task_diagnostics import finite_episode_returns
    plan=torch.load(Path(cli.workdir)/"plan.pt",map_location=cli.device,weights_only=False)
    alive=torch.ones(cli.num_envs,device=cli.device,dtype=torch.bool)
    for action in plan["prefix"]:
        result=session.step(action)
        alive &= ~result.terminated
        _reset_ended(session,result.terminated)
    initial_alive=alive.clone()
    start,context=session.capture()
    multiplier={"base":0.,"repeat":0.,"plus":1.,"minus":-1.}[cli.worker]
    native,imagined,tasks,auxes,terms,timeouts,successes=[],[],[],[],[],[],[]
    mismatches=torch.zeros(3,device=cli.device,dtype=torch.long)
    first_action=None
    with torch.no_grad():
        for step in range(cli.max_steps):
            previous,condition=session.capture()
            if step==0:
                latent=plan["mean"]+multiplier*cli.radius*plan["direction"]
                action=session.trainer.agents.policy.mean_to_action(latent)
            else:
                latent,action=_real_mean(cli,session,cli.prefix_steps+step)
            result=session.step(action)
            if first_action is None:first_action=result.executed_action.clone()
            parts=session.adapter.reward_components(previous,result.state,result.executed_action,condition)
            raw_cost=session.adapter.raw_action_cost(action,latent)
            native.append(result.reward)
            tasks.append(parts["task"])
            auxes.append(parts["weighted_aux"])
            imagined.append(parts["total"]-raw_cost)
            timeout=session.env.timeout_buf.bool().clone()
            terminal=result.terminated & ~timeout
            success=session.env.pick_success.bool().clone()
            terms.append(terminal);timeouts.append(timeout);successes.append(success & alive)
            outcome=session.adapter.task_outcome(previous,result.state,result.executed_action,condition)
            mismatches[0] += ((outcome["success"]!=success)&alive).sum()
            mismatches[1] += (((outcome["failed"]|outcome["completed"])!=terminal)&alive).sum()
            mismatches[2] += ((outcome["timeout"]!=timeout)&alive).sum()
            alive &= ~(terminal|timeout)
            _reset_ended(session,result.terminated)
            if not bool(alive.any()):break
        gamma=_teacher_gamma(session.trainer.agents)
        term,timeout=torch.stack(terms),torch.stack(timeouts)
        returns,complete,length=finite_episode_returns(torch.stack(imagined),term,timeout,gamma,initial_alive)
        native_return,_,_=finite_episode_returns(torch.stack(native),term,timeout,gamma,initial_alive)
        task_return,_,_=finite_episode_returns(torch.stack(tasks),term,timeout,gamma,initial_alive)
        aux_return,_,_=finite_episode_returns(torch.stack(auxes),term,timeout,gamma,initial_alive)
        record={"start":start,"context":context,"initial_alive":initial_alive,"complete":complete,
            "returns":returns,"native_returns":native_return,"task_returns":task_return,"aux_returns":aux_return,
            "length":length,"success":torch.stack(successes).any(0),"first_action":first_action,
            "families":session.probe_families,"predicate_mismatch_counts":mismatches,
            "family_names":session.probe_family_names,"gamma":gamma,"steps_run":len(native)}
        torch.save({key:value.detach().cpu() if torch.is_tensor(value) else value for key,value in record.items()},
                   Path(cli.workdir)/(cli.worker+".pt"))


def _worker(cli):
    # Required import order: no module imports torch before Isaac Gym.
    from isaacgym import gymapi  # noqa:F401
    import torch
    from probe_teacher_world_model import _load_bundle,_new_session,_file_sha256
    from utils.world_model.probes import seed_probe
    if not cli.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Real diagnostics require explicit CUDA simulation and execution")
    if _file_sha256(cli.checkpoint)!=cli.bundle_sha256:
        raise ValueError("Candidate bundle changed during independent diagnostic workers")
    torch.set_num_threads(1);seed_probe(cli.seed)
    bundle=_load_bundle(torch,cli.checkpoint)
    session=_new_session(cli,bundle,torch)
    try:
        if cli.worker=="plan":_plan(cli,session,bundle,torch)
        else:_real_branch(cli,session,bundle,torch)
    finally:
        writer=getattr(session.trainer.agents,"writer",None)
        if writer is not None:writer.close()
        session.close()


def _run(cli):
    import torch
    from probe_teacher_world_model import _load_bundle,_file_sha256
    from utils.world_model.task_diagnostics import mean_interval,value_diagnostic_report,paired_direction_report
    bundle=_load_bundle(torch,cli.checkpoint)
    if not bundle.get("imagined_critic"):
        raise ValueError("Candidate lacks matching imagined_critic; export a new snapshot before value diagnostics")
    bundle_hash=_file_sha256(cli.checkpoint)
    start=time.time();directory=Path(cli.workdir)
    for worker in ("plan","base","plus","minus","repeat"):
        command=[sys.executable,str(ROOT/"diagnose_manipulation_teacher.py"),"--checkpoint",cli.checkpoint,
            "--output",cli.output,"--num_envs",str(cli.num_envs),"--prefix_steps",str(cli.prefix_steps),
            "--max_steps",str(cli.max_steps),"--seed",str(cli.seed),"--device",cli.device,
            "--direction",cli.direction,"--coordinate",str(cli.coordinate),"--radius",str(cli.radius),
            "--family_sampling",cli.family_sampling,"--workdir",cli.workdir,
            "--worker",worker,"--bundle_sha256",bundle_hash]
        logfile=directory/(worker+".log")
        print("Launching independent diagnostic "+worker+": "+str(logfile),flush=True)
        environment=dict(os.environ)
        environment["PATH"]=str(Path(sys.prefix)/"bin")+os.pathsep+environment.get("PATH","")
        environment["LD_LIBRARY_PATH"]=str(Path(sys.prefix)/"lib")+os.pathsep+environment.get("LD_LIBRARY_PATH","")
        from utils.wm_entry import run_probe_subprocess
        with logfile.open("w") as stream:
            returncode=run_probe_subprocess(command,str(ROOT),stream,cli.worker_timeout,environment)
        if returncode:
            raise RuntimeError("Diagnostic worker %s failed; %s\n%s"%(worker,logfile,
                "\n".join(logfile.read_text(errors="replace").splitlines()[-30:])))
    plan=torch.load(directory/"plan.pt",map_location="cpu",weights_only=False)
    branches={name:torch.load(directory/(name+".pt"),map_location="cpu",weights_only=False)
              for name in ("base","plus","minus","repeat")}
    base=branches["base"]
    same_start=torch.ones(cli.num_envs,dtype=torch.bool)
    differences={}
    for name,branch in branches.items():
        error=(branch["start"]-base["start"]).abs().amax(-1)
        context_error=(branch["context"]-base["context"]).abs().amax(-1)
        # Report actual differences; do not round them away to fabricate pairs.
        same_start &= (error<=1e-6)&(context_error<=1e-6)&(branch["families"]==base["families"])
        differences[name]={"max_state":float(error.max()),"max_context":float(context_error.max())}
    plan_error=(plan["state"]-base["start"]).abs().amax(-1)
    plan_context_error=(plan["context"]-base["context"]).abs().amax(-1)
    same_start &= (plan_error<=1e-6)&(plan_context_error<=1e-6)&(plan["families"]==base["families"])
    differences["plan"]={"max_state":float(plan_error.max()),"max_context":float(plan_context_error.max())}
    initial=base["initial_alive"]&plan["initial_alive"]&same_start
    complete=initial.clone()
    for branch in branches.values():complete &= branch["complete"]&branch["initial_alive"]
    value=value_diagnostic_report(plan["initial_value"],base["returns"],base["complete"],initial)
    direction=paired_direction_report(base["returns"],branches["plus"]["returns"],branches["minus"]["returns"],
        branches["repeat"]["returns"],plan["model_plus"],plan["model_minus"],complete)
    groups={}
    for index,name in enumerate(base["family_names"]):
        mask=initial&(base["families"]==index)
        groups[name]={"value":value_diagnostic_report(plan["initial_value"],base["returns"],base["complete"],mask),
                      "success_rate_complete_cases":mean_interval(base["success"][mask&base["complete"]].float()),
                      "observed_success_count":int((base["success"]&mask).sum()),
                      "native_return":mean_interval(base["native_returns"][mask&base["complete"]])}
    reference=bundle["config"].get("recipe")=="reference"
    report={"schema":"dq-manipulation-independent-value-direction-v1","status":"diagnostic_not_training_benefit_evidence",
        "checkpoint":cli.checkpoint,"bundle_sha256":bundle_hash,"device":cli.device,"seed":cli.seed,
        "family_sampling":cli.family_sampling,"prefix_steps":cli.prefix_steps,"max_steps":cli.max_steps,
        "gamma":base["gamma"],"horizon":bundle["config"].get("horizon",4),
        "critic_updates":bundle.get("imagined_critic_updates",0),
        "observation_noise":("none" if float(bundle["config"].get("observation_noise_scale",0.))==0
                             else "training_distribution_common_step_seeds"),
        "return_target":("real_POST_reference_surrogate_minus_raw_latent_cost_until_native_end_no_bootstrap"
                         if reference else "real_POST_task_plus_weighted_aux_minus_raw_latent_cost_until_native_end_no_bootstrap"),
        "learning_algorithm":"reference_pure_bptt_v1" if reference else "legacy_ppo_plus_bptt",
        "model_coordinates":bundle.get("metadata",{}).get("model_coordinates"),
        "direction_space":"first_raw_deterministic_action" if reference else "first_latent_Gaussian_mean","direction_kind":cli.direction,
        "coordinate":cli.coordinate if cli.direction=="coordinate" else None,"radius":cli.radius,
        "start_repeatability":differences,"valid_start_pairs":int(initial.sum()),"value":value,"direction":direction,
        "families":groups,"predicate_mismatch_counts":base["predicate_mismatch_counts"].tolist(),
        "native_task_return_difference":mean_interval((base["task_returns"]-base["native_returns"])[initial&base["complete"]]),
        "aux_return":mean_interval(base["aux_returns"][initial&base["complete"]]),
        "executed_first_action_plus_minus_per_dimension":((branches["plus"]["first_action"]-branches["minus"]["first_action"])[initial].abs().mean(0).tolist() if bool(initial.any()) else None),
        "unmeasured_contact_state":"independent_same_seed_prefix_and_repeatability_not_PhysX_rewind",
        "elapsed_seconds":time.time()-start,"evidence_directory":cli.workdir}
    Path(cli.output).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":cli.output,"complete":value["complete"],"unfinished":value["unfinished"],
                      "value_error":value["error"],"direction":direction},ensure_ascii=False,indent=2))


def main():
    cli=parser().parse_args()
    if cli.num_envs<1 or cli.prefix_steps<0 or cli.max_steps<1 or not math.isfinite(cli.radius) or cli.radius<=0 or not math.isfinite(cli.worker_timeout) or cli.worker_timeout<=0:
        raise ValueError("Positive diagnostic size/radius/timeout and nonnegative prefix are required")
    cli.checkpoint=str(Path(cli.checkpoint).expanduser().resolve())
    cli.output=str(Path(cli.output).expanduser().resolve())
    Path(cli.output).parent.mkdir(parents=True,exist_ok=True)
    cli.workdir=str(Path(cli.workdir or cli.output+".artifacts").expanduser().resolve())
    Path(cli.workdir).mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT)
    if cli.worker is None:_run(cli)
    else:_worker(cli)


if __name__=="__main__":main()
