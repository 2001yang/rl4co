from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from rl4co.envs.common.base import RL4COEnvBase

from .generator import TSPDLGenerator

from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.data.transforms import StateAugmentation


def get_action_mask(td, pip_step=1, round_error_epsilon=1e-5):
    if pip_step == 0:
        load_on_arrival = td["current_load"].unsqueeze(-1) + td["demand"]
        meets_draft_limit = load_on_arrival <= (td["draft_limit"] + round_error_epsilon)
        unvisited = ~td["visited"]
        can_visit = unvisited & meets_draft_limit

    elif pip_step == 1:
        load_succ = td["current_load"].unsqueeze(-1) + td["demand"]  # [B, n]
        load_grandsucc = load_succ.unsqueeze(-1) + td["demand"].unsqueeze(1)  # [B, n, n]

        succ_feasible = load_succ <= (td["draft_limit"] + round_error_epsilon)  # [B, n]
        grandsucc_feasible = load_grandsucc <= (td["draft_limit"].unsqueeze(1) + round_error_epsilon)  # [B, n, n]

        eye = torch.eye(td["locs"].size(1), dtype=torch.bool, device=td.device).unsqueeze(0)
        skip_mask = td["visited"].unsqueeze(1) | eye  # [B, n, n]
        grandsucc_check = (grandsucc_feasible | skip_mask).all(dim=-1)

        unvisited = ~td["visited"]
        can_visit = unvisited & succ_feasible & grandsucc_check  # [B, n]

    return can_visit


class TSPDLEnv(RL4COEnvBase):
    def __init__(self, generator=TSPDLGenerator, generator_params={}, **kwargs):
        self.pip_step = kwargs.pop("pip_step", 1)
        super().__init__(check_solution=False, **kwargs)
        self.generator = generator(**generator_params)
        self.round_error_epsilon = 1e-5

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        visited = torch.zeros((*batch_size, td["locs"].size(1)), dtype=torch.bool, device=td.device)
        visited[:, 0] = True

        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "demand": td["demand"],
                "draft_limit": td["draft_limit"],
                "current_node": torch.zeros(*batch_size, dtype=torch.int64, device=td.device),
                "current_load": torch.zeros(*batch_size, dtype=torch.float32, device=td.device),
                "draft_limit_violation": torch.zeros_like(visited, dtype=torch.float32),
                "visited": visited,
            },
            batch_size=td.batch_size,
            device=td.device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_action_mask(self, td):
        unvisited = ~td["visited"]
        can_visit = get_action_mask(td, pip_step=self.pip_step, round_error_epsilon=self.round_error_epsilon)  # [B, n]
        action_mask = torch.where(can_visit.any(-1, keepdim=True), can_visit, unvisited)
        return action_mask

    def _step(self, td) -> TensorDict:
        batch_idx = torch.arange(td["locs"].size(0), device=td.device)
        current_node = td["action"]
        current_load = td["current_load"] + gather_by_index(td["demand"], current_node)
        current_draft_limit = gather_by_index(td["draft_limit"], current_node)
        td["draft_limit_violation"][batch_idx, current_node] = (current_load - current_draft_limit).clamp_(min=0.0)

        visited = td["visited"].scatter_(1, current_node.unsqueeze(1), 1)
        done = visited.sum(1) == visited.size(1)
        reward = torch.zeros_like(done, dtype=torch.float32)
        td.update(
            {
                "visited": visited,
                "current_node": current_node,
                "current_load": current_load,
                "reward": reward,
                "done": done,
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

        draft_limit_viol = td["draft_limit_violation"]  # [B, n]
        total_constraint_violation = draft_limit_viol.sum(dim=1)  # [B]
        violated_node_count = (draft_limit_viol > self.round_error_epsilon).sum(dim=1).float()  # [B]

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
