from typing import Optional

import torch
from torch import Tensor

from tensordict.tensordict import TensorDict
from rl4co.envs.common.base import RL4COEnvBase

from .generator import TSPTWGenerator

from rl4co.utils.ops import get_distance, gather_by_index, unbatchify
from rl4co.data.transforms import StateAugmentation

class TSPTWEnv(RL4COEnvBase):
    """Traveling Salesman Problem with Time Window constraints (TSPTW) environment
    At each step, the agent chooses a city to visit. The reward is 0 unless the agent visits all the cities.
    In that case, the reward is (-)length of the path + (-) time window violations + (-) total count of the violated nodes.
    Refer to Bi, et al., 2024 for more details (https://arxiv.org/abs/2410.21066).

    Observations:
        - locations of each customer.
        - time window of each customer.
        - the current location of the vehicle.
        - the current time.

    Constraints:
        - the tour must return to the starting customer.
        - each customer must be visited exactly once.
        - the arrival time of each customer must be within the time window.
        (Note that the time window is unable to enforced as the mask is NP-hard; Refer to Bi, et al., 2024)

    Finish condition:
        - the agent has visited all customers and returned to the starting customer.

    Reward:
        - (minus) the negative length of the path.
        - (minus) the negative summation of time window violations.
        - (minus) the negative total count of the violated nodes.

    Args:
        generator: TSPTWGenerator instance as the data generator
        generator_params: parameters for the generator
    """

    name = "tsptw"
    def __init__(
            self,
            generator:TSPTWGenerator = None, 
            generator_params: dict = {}, 
            **kwargs):
        self.pip_step = kwargs.pop("pip_step", 1)
        super().__init__(check_solution=False, **kwargs)
        if generator is None:
            generator = TSPTWGenerator(**generator_params)
        self.generator = generator
        self.round_error_epsilon = 1e-5

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        device = td.device
        num_locs = td["locs"].size(1)

        visited = torch.zeros((*batch_size, num_locs), dtype=torch.bool, device=device)
        visited[:, 0] = True

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "service_time": td["service_time"],
                "time_windows": td["time_windows"],
                "current_time": torch.zeros(*batch_size, dtype=torch.float32, device=device),
                "current_node": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "service_start_time_cache": torch.zeros_like(visited),
                "action": torch.zeros(*batch_size, dtype=torch.int64, device=device),
                "visited": visited,
                "time_window_violation": torch.zeros((*batch_size, num_locs), dtype=torch.float32, device=device),
            },
            batch_size=batch_size,
            device=device,
        )
        if self.pip_step == 1:
            td_reset["distance_matrix"] = torch.cdist(td["locs"], td["locs"], p=2, compute_mode="donot_use_mm_for_euclid_dist")
            td_reset["duration_matrix"] = td_reset["distance_matrix"] + td["service_time"][:, :, None]
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset
    
    def get_action_mask(self, td):
        unvisited = ~td["visited"]
        can_visit = self.get_pip_mask(td, pip_step=self.pip_step, round_error_epsilon=self.round_error_epsilon)
        action_mask = torch.where(can_visit.any(-1, keepdim=True), can_visit, unvisited)
        return action_mask

    def _step(self, td) -> TensorDict:
        """
        update the state of the environment, including
        current_node, current_time, time_window_violation, visited and action_mask
        """
        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        prev_node, curr_node = td["current_node"], td["action"]
        prev_loc, curr_loc = (
            td["locs"][batch_idx, prev_node],
            td["locs"][batch_idx, curr_node],
        ) 

        travel_time = get_distance(prev_loc, curr_loc) 

        arrival_time = td["current_time"] + travel_time
        tw_start_curr, tw_end_curr = (td["time_windows"][batch_idx, curr_node]).unbind(-1)
        service_time = td["service_time"][batch_idx, curr_node]
        curr_time = torch.max(arrival_time, tw_start_curr) + service_time
        td["time_window_violation"][batch_idx, curr_node] = torch.clamp(arrival_time - tw_end_curr, min=0.0)

        visited = td["visited"].scatter_(1, curr_node[..., None], True)
        done = visited.sum(dim=-1) == visited.size(-1)
        reward = torch.zeros_like(done, dtype=torch.float32)

        td.update(
            {
                "current_time": curr_time,
                "current_node": curr_node,
                "visited": visited,
                "done": done,
                "reward": reward,
            }
        )
        num_unvisited = (~td["visited"][0]).sum().item()
        action_mask = self.get_action_mask(td) if num_unvisited > 1 else ~visited

        td.set("action_mask", action_mask)

        return td

    def _get_reward(self, td, actions):
        ordered_locs = torch.cat([td["locs"][:, 0:1], gather_by_index(td["locs"], actions)], dim=1)
        diff = ordered_locs - ordered_locs.roll(-1, dims=-2)
        tour_length = diff.norm(dim=-1).sum(-1)

        tw_viol = td["time_window_violation"]  # [B, n]
        total_constraint_violation = tw_viol.sum(dim=1)  # [B]
        violated_node_count = (tw_viol > self.round_error_epsilon).sum(dim=1).float()  # [B]

        return TensorDict(
            {
                "negative_length": -tour_length,
                "total_constraint_violation": total_constraint_violation,
                "violated_node_count": violated_node_count,
            },
            batch_size=td["locs"].size(0),
            device=td.device,
        )

    def rollout(self, td, policy, num_samples=1, decode_type="greedy", device="cuda"):
        with torch.inference_mode():
            with torch.amp.autocast("cuda") if "cuda" in str(device) else torch.inference_mode():
                td = td.to(device)
                td = self.reset(td)
                td_aug = StateAugmentation(num_augment=8, augment_fn="dihedral8")(td)
                num_samples = 0 if decode_type == "greedy" else num_samples

                out = policy(td_aug, self, decode_type=decode_type, num_samples=num_samples)
                actions = unbatchify(out["actions"], (8, num_samples))

                reward_td = unbatchify(out["reward"], (8, num_samples))
                reward, total_constraint_violation = (
                    reward_td["negative_length"],
                    reward_td["total_constraint_violation"],
                )
                sol_feas = total_constraint_violation < self.round_error_epsilon

        return TensorDict(
            {
                "actions": actions,
                "reward": reward,
                "sol_feas": sol_feas,
            },
            batch_size=td.batch_size[0],
            device=td.device,
        )

    def get_penalized_reward(self, td, actions, rho_c=1.0, rho_n=1.0, return_dict=False):
        ordered_locs = torch.cat([td["locs"][:, 0:1], gather_by_index(td["locs"], actions)], dim=1)
        diff = ordered_locs - ordered_locs.roll(-1, dims=-2)
        tour_length = diff.norm(dim=-1).sum(-1)
        tw_viol = self.get_time_window_violations(td, actions)
        total_constraint_violation = tw_viol.sum(dim=1)
        violated_node_count = (tw_viol > self.round_error_epsilon).sum(dim=1).float()
        penalized_obj = tour_length + rho_c * total_constraint_violation + rho_n * violated_node_count

        if return_dict:
            return TensorDict(
                {
                    "negative_length": -tour_length,
                    "total_constraint_violation": total_constraint_violation,
                    "violated_node_count": violated_node_count,
                },
                batch_size=td["locs"].size(0),
                device=td.device,
            )
        return -penalized_obj

    def get_pip_mask(self, td, pip_step=1, round_error_epsilon=1e-5):
        if pip_step == 0:
            curr_node = td["current_node"]

            # time window constraint
            d_ij = get_distance(gather_by_index(td["locs"], curr_node)[:, None, :], td["locs"])  # [B, n]
            arrival_time = td["current_time"][:, None] + d_ij
            can_reach_in_time = arrival_time <= (td["time_windows"][..., 1] + round_error_epsilon)  # [B, n]

            unvisited = ~td["visited"]

            can_visit = unvisited & can_reach_in_time  # [B, n]

        elif pip_step == 1:
            batch_size, num_locs, _ = td["locs"].shape
            batch_idx = torch.arange(batch_size, device=td.device)  # [B, ]

            tw_start, tw_end = td["time_windows"].unbind(-1)

            dur_cur_succ = td["duration_matrix"][batch_idx, td["current_node"], :]

            service_start_time_succ = torch.max(td["current_time"].unsqueeze(1) + dur_cur_succ, tw_start)
            service_start_time_grandsucc = torch.max(service_start_time_succ.unsqueeze(-1) + td["duration_matrix"], tw_start.unsqueeze(1))

            succ_feasible = service_start_time_succ <= (tw_end + round_error_epsilon)  # [B, n]
            grandsucc_feasible = service_start_time_grandsucc <= (tw_end.unsqueeze(1) + round_error_epsilon)  # [B, n, n]

            eye = torch.eye(num_locs, dtype=torch.bool, device=td.device).unsqueeze(0)
            skip_mask = td["visited"].unsqueeze(1) | eye  # [B, n, n]
            grandsucc_check = (grandsucc_feasible | skip_mask).all(dim=-1)

            unvisited = ~td["visited"]
            can_visit = unvisited & succ_feasible & grandsucc_check  # [B, n]

        elif pip_step == -1:
            unvisited = ~td["visited"]
            can_visit = unvisited  # [B, n]

        return can_visit

    def get_time_window_violations(self, td, actions: Tensor) -> Tensor:
        """
        Optimized vectorized implementation, eliminating explicit loops
        Args:
            td: TensorDict containing:
                - time_windows: [B, n, 2]
                - duration_matrix: [B, n, n]
            actions: [B, n-1] customer node permutation
        Returns:
            violations: [B, n] time window violation
        """
        batch_size, n = actions.shape
        device = actions.device

        paths = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device=device), actions], dim=1)

        time_windows = td["time_windows"].gather(1, paths.unsqueeze(-1).expand(-1, -1, 2))
        tw_start, tw_end = time_windows.unbind(-1)

        batch_idx = torch.arange(batch_size, device=device)[:, None]
        durations = td["duration_matrix"][batch_idx, paths[:, :-1], paths[:, 1:]]

        service_start = torch.empty(batch_size, n + 1, device=device)
        service_start[:, 0] = torch.maximum(torch.zeros(batch_size, device=device), tw_start[:, 0])

        for step in range(1, n + 1):
            arrival_time = service_start[:, step - 1] + durations[:, step - 1]
            service_start[:, step] = torch.maximum(arrival_time, tw_start[:, step])

        time_window_violation = torch.clamp_min(service_start - tw_end, 0)
        return time_window_violation