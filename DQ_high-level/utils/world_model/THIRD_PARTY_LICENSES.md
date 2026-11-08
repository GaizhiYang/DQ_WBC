# SuperMimic-Reproduced attribution

The residual MLP, zero initialization, GPU replay organization, parameter-frozen
model differentiation, and autoregressive fitting in this package are adapted
from the local independent reproduction `/home/hehui/SuperMimic-Reproduced`,
commit `ba6105349069bf040f1fdad60873d13138d505a2`:

- `src/supermimic/world_model/dynamics.py`
- `src/supermimic/world_model/state_spec.py`
- `src/supermimic/world_model/world_model.py`
- `src/supermimic/world_model/wm_trainer.py`
- `src/supermimic/data/replay_buffer.py`
- `src/supermimic/imagination/policy_trainer.py`
- `src/supermimic/imagination/objective.py`
- `src/supermimic/imagination/rollout.py`
- `src/supermimic/policy/nets.py` (imagined critic and small Actor output initialization)
- `src/supermimic/envs/mdp/rewards.py` (all-coordinate squared action-rate cost)
- `src/supermimic/envs/actions.py` (normalized command followed by physical target scaling)
- `src/supermimic/kinematics/rotations.py` (rot6d and rotation log error)
- `src/supermimic/kinematics/g1_fk.py` (forward position/velocity recursion)

The DQ adapter, physical state layout, episode-held-out replay, PPO integration,
and task-specific gradient restrictions are new adaptations. This is not a copy
of official SuperMimic source, and it does not depend on MuJoCo Warp.

## Original license

MIT License

Copyright (c) 2026 Yuxi Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Empirical Actor observation normalization (RSL-RL)

`reference_policy.py:EmpiricalObservationNorm` adapts the explicit running-moment
equations and normalization denominator from
`rsl_rl/modules/normalization.py:EmpiricalNormalization`, as used by the local
`supermimic_manipulation` reference. Forward evaluation preserves input gradients;
DQ adds finite-input checks and updates only from real returned observations.
The upstream source header additionally credits Copyright (c) 2020 Preferred
Networks, Inc., and Copyright (c) 2021-2026 ETH Zurich and NVIDIA CORPORATION.

Copyright (c) 2026, ETH Zurich
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Manipulation adaptation reference

The explicit RunningNorm statistics, neural-only BF16 autocast, masked rollout
fitting, parameter freezing and TD-lambda/value preparation organization also
reference `/home/hehui/supermimic_manipulation`, commit
`eda7da480c6d92a68cc41c9628b7efdae08cae23`:

- `src/clab/algorithms/supermimic/networks.py` (RunningNorm and MLP autocast)
- `src/clab/algorithms/supermimic/agent.py` (frozen imagined rollout and fixed value labels)
- `src/clab/algorithms/supermimic/buffer.py` (terminal-prefix validity)
- `src/clab/tasks/supermimic.py` (straight-through predicates, analytic success
  counting, gradient-scaled success, task-owned reward and critic inputs)
- `src/clab/tasks/lemon/supermimic.py` and `box_in_bin/supermimic.py` (explicit
  endpoint task reward, action/velocity costs and success-derived survival)
- `docs/supermimic_manipulation_training_flow.html` (training responsibilities)

DQ's W/V/B boundary budgets, residual-target inverse, continuous-gripper
geometry objective and fixed-eight low-controller loop are task adaptations,
not source-equivalent hyperparameters or a proven manipulation reward.
The newer `reference_task.py` follows the listed task-MDP structure while
retaining DQ's native height/distance/continuous-opening success predicates.
Its bounded lift and relative object/tool slip holding rewards adapt the
source's placement/release objectives to mobile picking; they do not model
contact forces and do not change native simulation success labels. The full
reference Critic includes causal rollout state and reconstructed observations;
it excludes the inherited stale end-effector cache because fixed8 uses FK.
The reference's `REFERENCE_LICENSE` carries the Yuxi Liu MIT notice reproduced
above. Its project LICENSE is reproduced below.

MIT License

Copyright (c) 2026 emoPointer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
