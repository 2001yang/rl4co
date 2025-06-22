from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler

class TSPTWGenerator(Generator):
    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 100.0,
        max_time: float = 100.0,
        hardness: str = "hard",
        normalize: bool = True,
        loc_distribution: int | float | str | type | Callable = Uniform,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.max_time = max_time
        self.hardness = hardness
        self.normalize = normalize
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        #sample locations
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))

        #sample hard time window
        if self.hardness == "hard":
            rand_solution = torch.stack([torch.randperm(self.num_loc - 1) + 1 for _ in range(*batch_size)])
            rand_solution = torch.cat([torch.zeros((*batch_size, 1), dtype=torch.long), rand_solution], dim=-1)

            rand_solution_locs = locs.gather(1, rand_solution.unsqueeze(-1).expand(-1, -1, 2))
            difference = rand_solution_locs[:, :-1] - rand_solution_locs[:, 1:]  
            distance = torch.norm(difference, dim=-1)  
            arrival_time = torch.cumsum(distance, dim=-1)

            tw_start_solution = torch.clamp(arrival_time - torch.rand((*batch_size, self.num_loc - 1)) * self.max_time / 2, min=0)
            tw_start_solution = torch.cat([torch.zeros((*batch_size, 1), dtype=torch.long), tw_start_solution], dim=-1)
            tw_end_solution = arrival_time + torch.rand((*batch_size, self.num_loc - 1)) * self.max_time / 2
            depot_tw_end = torch.full((*batch_size, 1), 1000. * self.max_time)
            tw_end_solution = torch.cat([depot_tw_end, tw_end_solution], dim=-1)

            tw_start, tw_end = torch.zeros_like(tw_start_solution), torch.zeros_like(tw_end_solution)
            tw_start.scatter_(1, rand_solution, tw_start_solution)
            tw_end.scatter_(1, rand_solution, tw_end_solution)
            
        #sample easy & medium time window
        elif self.hardness in ["easy", "medium"]:
            tw_start, tw_end = self.generate_time_windows(
                locs, self.num_loc * 55, 0.5 if self.hardness == "easy" else 0.1, 0.75 if self.hardness == "easy" else 0.2
            )

        if self.normalize:
            scaler = self.max_loc - self.min_loc
            locs, tw_start, tw_end = locs / scaler, tw_start / scaler, tw_end / scaler

        return TensorDict(
            {
                "locs": locs,
                "time_windows": torch.stack([tw_start, tw_end], dim=-1),
                "service_time": torch.zeros((*batch_size, self.num_loc), dtype=torch.float32),
            },
            batch_size=batch_size,
        )

    def generate_time_windows(self, locs, time_factor, alpha, beta):
        batch_size, num_loc = locs.size(0), locs.size(1)  

        tw_start = torch.randint(0, time_factor, (batch_size, num_loc))
        tw_start[:, 0] = 0

        epsilon = torch.rand(batch_size, num_loc) * (beta - alpha) + alpha
        duration = torch.round(time_factor * epsilon)
        duration[:, 0] = time_factor * 2

        tw_end = tw_start + duration
        tw_end = tw_end.type(torch.long)

        return tw_start, tw_end