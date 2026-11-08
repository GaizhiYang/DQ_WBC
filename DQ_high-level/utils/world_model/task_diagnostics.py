"""Pure tensor return/noise/statistics helpers for independent real diagnostics."""

from contextlib import contextmanager
import math

import torch


@contextmanager
def diagnostic_noise_seed(seed, device):
    """Common observation noise without consuming the simulator's RNG stream."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(int(seed))
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(int(seed))
        yield


def diagnostic_policy_mean(policy, preprocessor, observation, *, noise=None, seed=0):
    """Training sensor noise, fixed preprocessing, deterministic Gaussian mean.

    The caller selects grad/no_grad. The noise module's configured train/eval
    mode is retained; no Gaussian action sample is drawn and no scaler updates.
    """
    with diagnostic_noise_seed(seed, observation.device):
        observation = noise(observation) if noise is not None else observation
    if preprocessor is not None:
        try:
            observation = preprocessor(observation, train=False, no_grad=False)
        except TypeError:
            observation = preprocessor(observation, train=False)
    mean, _, _ = policy.compute({"states": observation}, role="policy")
    return mean, policy.mean_to_action(mean)


def finite_episode_returns(rewards, terminated, timeouts, gamma=.99, initial_alive=None):
    """Keep the first terminal/timeout reward, never bootstrap or cross reset.

    Returns (discounted return, completed, valid transition count). Unfinished
    rows remain censored: their partial return is not a value ground truth.
    """
    if rewards.ndim != 2 or rewards.shape[0] < 1 or terminated.shape != rewards.shape or timeouts.shape != rewards.shape:
        raise ValueError("Episode diagnostics require matching [time,batch] tensors")
    if not 0 < gamma <= 1:
        raise ValueError("Diagnostic gamma must lie in (0,1]")
    alive = torch.ones(rewards.shape[1],device=rewards.device,dtype=torch.bool) if initial_alive is None else initial_alive.bool().clone()
    initial = alive.clone()
    total = torch.zeros_like(rewards[0])
    length = torch.zeros_like(alive,dtype=torch.long)
    for step in range(len(rewards)):
        selected = torch.where(alive,rewards[step],torch.zeros_like(rewards[step]))
        if not bool(torch.isfinite(selected).all()):
            raise FloatingPointError("Nonfinite reward on a live diagnostic transition")
        total = total + gamma**step * selected
        length += alive.long()
        alive = alive & ~(terminated[step].bool() | timeouts[step].bool())
    return total, initial & ~alive, length


def mean_interval(values):
    """Descriptive normal-approximation 95% interval; not a success threshold."""
    values=values.detach().double().cpu().reshape(-1)
    finite=torch.isfinite(values)
    values=values[finite]
    n=len(values)
    if not n:
        return {"n":0,"mean":None,"ci95":None,"nonfinite":int((~finite).sum())}
    mean=float(values.mean())
    width=1.96*float(values.std(unbiased=True))/math.sqrt(n) if n>1 else None
    return {"n":n,"mean":mean,"ci95":[mean-width,mean+width] if width is not None else None,
            "nonfinite":int((~finite).sum()),"interval":"normal_approximation_across_initial_states"}


def value_diagnostic_report(prediction, real_returns, completed, initial_alive):
    valid=completed.bool() & initial_alive.bool() & torch.isfinite(prediction) & torch.isfinite(real_returns)
    predicted,actual=prediction[valid].double(),real_returns[valid].double()
    error=predicted-actual
    correlation=None
    if len(error)>1 and bool(predicted.std()>0) and bool(actual.std()>0):
        correlation=float(torch.corrcoef(torch.stack((predicted,actual)))[0,1])
    # Pairwise value ordering, ignoring exact return ties. Cap deterministically
    # for a diagnostic launched with thousands of independent environments.
    predicted,actual=predicted[:512],actual[:512]
    dp,dr=predicted[:,None]-predicted[None,:],actual[:,None]-actual[None,:]
    pairs=torch.triu(torch.ones_like(dp,dtype=torch.bool),diagonal=1)&(dr!=0)
    ordering=float(((dp*dr)>0)[pairs].float().mean()) if bool(pairs.any()) else None
    return {"eligible":int(initial_alive.sum()),"complete":int(valid.sum()),
            "unfinished":int((initial_alive.bool()&~completed.bool()).sum()),
            "nonfinite_prediction_or_return":int((initial_alive.bool() &
                (~torch.isfinite(prediction) | ~torch.isfinite(real_returns))).sum()),
            "error":mean_interval(error),"absolute_error":mean_interval(error.abs()),
            "rmse":float(error.square().mean().sqrt()) if len(error) else None,
            "pearson":correlation,"pairwise_order_agreement":ordering,
            "order_rows":len(predicted),"status":"diagnostic_only_thresholds_uncalibrated",
            "unfinished_status":"not_accepted_as_value_ground_truth"}


def paired_direction_report(base,plus,minus,repeat,model_plus,model_minus,valid):
    finite=torch.isfinite(torch.stack((base,plus,minus,repeat,model_plus,model_minus))).all(0)
    valid=valid.bool()&finite
    effect=(plus-minus)[valid]
    predicted=(model_plus-model_minus)[valid]
    noise=(repeat-base)[valid].abs()
    informative=(effect.abs()>2*noise)&(predicted.abs()>0)
    return {"pairs":int(valid.sum()),"real_plus_minus":mean_interval(effect),
            "real_plus_base":mean_interval((plus-base)[valid]),
            "model_plus_minus":mean_interval(predicted),"base_repeat_absolute":mean_interval(noise),
            "informative_pairs":int(informative.sum()),
            "direction_agreement":float(((effect*predicted)>0)[informative].float().mean()) if bool(informative.any()) else None,
            "model_improves_real_worsens":int(((predicted>0)&(effect<0)&informative).sum()),
            "status":"diagnostic_only_not_exact_contact_derivative"}
