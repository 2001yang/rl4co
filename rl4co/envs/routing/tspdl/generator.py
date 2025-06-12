from typing import Callable

import torch
import numpy as np

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler

class TSPDLGenerator(Generator):
    def __init__(
        self, 
        num_loc: int = 50, 
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        hardness: str = "hard", 
        draft_method: str = "rejection", 
        normalize: bool = True, 
        loc_distribution: int | float | str | type | Callable = Uniform,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.hardness = hardness
        self.draft_method = draft_method
        self.normalize = normalize
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))

        demand = torch.cat(
            [torch.zeros((*batch_size, 1)), torch.ones((*batch_size, self.num_loc - 1))],
            dim=-1,
        )

        # Sum of port demand
        demand_sum = self.num_loc - 1
        # Draft limit generation
        draft_limit = self._generate_draft_limit(batch_size, demand_sum)

        if self.normalize:
            demand = demand / demand_sum
            draft_limit = draft_limit / demand_sum

        return TensorDict(
            {
                "locs": locs, 
                "demand": demand, 
                "draft_limit": draft_limit
            },
            batch_size=batch_size,
        )

    def _generate_draft_limit(self, batch_size, demand_sum):
        constrain_pct = {"hard": 0.9, "medium": 0.75, "easy": 0.5}[self.hardness]
        num_constrained = int(constrain_pct * (self.num_loc))

        # Initialize all limits to demand_sum
        draft_limit = torch.full((*batch_size, self.num_loc), demand_sum, dtype=torch.float)

        # Batch processing with numpy compatibility
        for i in range(*batch_size):
            # randomly choose w% of the nodes (except depot) to lower their draft limit (range: [1, demand_sum))
            lower_dl_idx = np.random.choice(range(1, self.num_loc), num_constrained, replace=False)
            feasible_dl = False
            while not feasible_dl:
                constrained_limits = torch.randint(1, demand_sum, (num_constrained,))
                cnt = torch.bincount(constrained_limits, minlength=demand_sum + 1)
                cnt_cumsum = torch.cumsum(cnt, dim=0)
                feasible_dl = (cnt_cumsum <= torch.arange(cnt_cumsum.size(0))).all()

            draft_limit[i][lower_dl_idx] = constrained_limits.float()

        return draft_limit