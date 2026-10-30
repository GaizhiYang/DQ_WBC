TSIL source attribution
=======================

The TSIL algorithms adapted in `DQ_high-level/utils/tsil.py` and
`DQ_high-level/utils/tsil_ppo.py` are based on the local TSIL source by
Yinsen Jia and Boyuan Chen, commit
`137e3ab8d6f1eccf9cf6c611f957d106c2ff000c`:

- `core/training/algo/tsil/sampling.py`: fastest-success-first selection and
  high-return fallback with duplicate removal.
- `core/training/algo/tsil/memory.py`: fastest/return ranking and trajectory returns.
- `core/training/algo/tsil/loss.py`: positive-gap actor/critic losses and
  positive actor-weight population standard deviation and cap.
- `core/training/algo/updater.py`: joint PPO/self-imitation optimization.

The upstream project distributes this code under Apache License 2.0, reproduced
in LICENSE. These adaptations change configuration identity, storage layout,
episode recording, normalization, and full-configuration replay sampling for
DQ's SGS resets and causal EKF observations. They do not import the upstream
Isaac Gym task framework. The paper-defined once-per-update positive-advantage
trajectory sampling is implemented explicitly where upstream differs.
