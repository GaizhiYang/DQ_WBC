"""Recent completed-episode outcomes for monitoring, independent of sampling.

These buffers never select resets, consume random numbers, or update training
state. Call ``record`` once per batch of naturally completed episodes, before
replacing their reset-family IDs. Aborted or still-running episodes must not be
submitted. Within a batch, the supplied order breaks simultaneous-finish ties.
"""

import torch


class RecentEpisodeStatistics:
    """Keep one separate recent-episode window for each reset family.

    Counts, successes, histories, and write pointers all stay on ``device``.
    A window with fewer than its capacity uses only its observed episodes;
    ``count == 0`` means no estimate, not an observed zero-percent success rate.
    """

    def __init__(self, num_families, device, family_window=100):
        for name, value in (("num_families", num_families),
                            ("family_window", family_window)):
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError("{} must be a positive integer".format(name))
        self.num_families = int(num_families)
        self.device = torch.device(device)
        self.family_window = int(family_window)
        self.family_history = torch.zeros(
            self.num_families, self.family_window, dtype=torch.bool, device=self.device)
        self.family_count = torch.zeros(self.num_families, dtype=torch.long, device=self.device)
        self.family_successes = torch.zeros_like(self.family_count)
        self.family_head = torch.zeros_like(self.family_count)

    @torch.no_grad()
    def record(self, family_ids, successes):
        """Append completed episodes in input order without any host scalar reads."""
        ids = torch.as_tensor(family_ids, device=self.device)
        outcomes = torch.as_tensor(successes, device=self.device)
        if ids.ndim != 1 or outcomes.ndim != 1 or outcomes.shape != ids.shape:
            raise ValueError("family_ids and successes must be matching one-dimensional tensors")
        if ids.numel() and ids.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("family_ids must contain integer IDs")
        if not ids.numel():
            return
        ids = ids.to(torch.long)
        # Async assertions avoid copying valid CUDA input scalars back to the
        # CPU. Invalid IDs/outcomes are caller errors, just as for tensor indexing.
        torch._assert_async(((ids >= 0) & (ids < self.num_families)).all(),
                            "family ID outside the configured family count")
        torch._assert_async(((outcomes == 0) | (outcomes == 1)).all(),
                            "episode outcomes must be Boolean or 0/1")
        outcomes = outcomes.to(torch.bool)

        # Stable sorting preserves completion order independently inside each
        # family. Discard an oversized batch's old prefix before indexed writes
        # so every destination occurs at most once, including on CUDA.
        order = torch.argsort(ids, stable=True)
        sorted_ids = ids[order]
        positions = torch.arange(ids.numel(), device=self.device)
        starts = torch.ones_like(sorted_ids, dtype=torch.bool)
        starts[1:] = sorted_ids[1:] != sorted_ids[:-1]
        group_start = torch.cummax(
            torch.where(starts, positions, torch.zeros_like(positions)), dim=0).values
        rank = positions - group_start
        additions = torch.zeros_like(self.family_count)
        additions.scatter_add_(0, sorted_ids, torch.ones_like(sorted_ids))
        keep = rank >= additions[sorted_ids] - self.family_window
        kept_ids = sorted_ids[keep]
        slots = (self.family_head[kept_ids] + rank[keep]) % self.family_window
        new_values = outcomes[order][keep]
        old_values = self.family_history[kept_ids, slots]
        self.family_successes.scatter_add_(0, kept_ids, new_values.long() - old_values.long())
        self.family_history[kept_ids, slots] = new_values
        self.family_count.add_(additions).clamp_(max=self.family_window)
        self.family_head.add_(additions).remainder_(self.family_window)

    def family_rates(self):
        """Return device rates; callers should use family_count to identify N/A."""
        return self.family_successes.float() / self.family_count.clamp_min(1)
