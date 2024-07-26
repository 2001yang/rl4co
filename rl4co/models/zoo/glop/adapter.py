from typing import Any, Generator, NamedTuple, Optional

import numba as nb
import numpy as np
import torch

from tensordict import TensorDict


class SubTSPMapping(NamedTuple):
    map_action_index: np.ndarray
    map_node_index: np.ndarray
    subtsp_coordinates: np.ndarray


class SubTSPAdapter(object):
    """TODO"""

    def __init__(self, parent_td: TensorDict, actions: torch.Tensor, min_node_count: int = 4) -> None:
        self.batch_size = parent_td.batch_size[0]
        self.n_samples = actions.shape[0] // self.batch_size
        assert actions.shape[0] == self.n_samples * self.batch_size

        # prepend depot to each route
        self._actions = np.concatenate([
            np.zeros((actions.shape[0], 1), dtype = np.int64),
            actions.cpu().numpy(),
        ], axis=1)

        self.map_action_index = _cvrp_action_partitioner(self._actions, min_node_count)
        self.coordinates = parent_td["locs"].cpu().numpy()

    def get_all_subtsps(self) -> SubTSPMapping:
        map_node_index, subtsp_coordinates = _compose_subtsp_coordinates(
            self._actions, self.map_action_index, self.coordinates
        )
        return SubTSPMapping(self.map_action_index, map_node_index, subtsp_coordinates)
    
    def get_actions(self):
        return torch.from_numpy(self._actions)

    def get_batched_subtsps(
        self, batch_size: Optional[int] = None
    ) -> Generator[SubTSPMapping, Any, None]:
        if batch_size is None:  # fallback to single batch
            yield self.get_all_subtsps()
        else:
            # group sub problems by node count
            node_count = self.map_action_index[:, 2] - self.map_action_index[:, 1]
            order = np.argsort(node_count)
            for start_index in range(0, len(order), batch_size):
                selected_subtsp_index = order[start_index : start_index + batch_size]
                map_action_index = self.map_action_index[selected_subtsp_index]
                map_node_index, subtsp_coordinates = _compose_subtsp_coordinates(
                    self._actions, map_action_index, self.coordinates
                )
                yield SubTSPMapping(map_action_index, map_node_index, subtsp_coordinates)

    def update_actions(self, mapping: SubTSPMapping, subtsp_actions: np.ndarray):
        assert subtsp_actions.shape == mapping.map_node_index.shape
        _update_cvrp_actions(
            self._actions,
            subtsp_actions,
            mapping.map_action_index,
            mapping.map_node_index,
        )

    @staticmethod
    def pre_compile_numba():
        # Use a tiny example to invoke numba's compilation
        # This might take around 16 seconds, depending on the CPU performance
        actions = np.array([[0, 1, 2, 3, 0, 4]], dtype=np.int64)
        coordinates = np.zeros((1, 5, 2), dtype=np.float32)
        subtsp_actions = np.array([[1, 2, 0, 4, 3]], dtype=np.int64)
        map_action_index = _cvrp_action_partitioner(actions, min_node_count=4)
        map_node_index, _ = _compose_subtsp_coordinates(
            actions, map_action_index, coordinates
        )
        _update_cvrp_actions(actions, subtsp_actions, map_action_index, map_node_index)
        # `actions` should be [[0,3,1,2,0,4]] at this moment


@nb.njit(nogil=True, parallel=False, cache=True)
def _cvrp_action_partitioner(routes: np.ndarray, min_node_count: int = 4):
    map_action_index = []
    route_length = routes.shape[1]
    for index in range(routes.shape[0]):
        start = 0
        last_is_not_zero = routes[index, 0] != 0
        for idx in range(1, route_length):
            node = routes[index, idx]
            if node == 0:
                if last_is_not_zero and idx - start >= min_node_count:
                    map_action_index.append((index, start, idx))
                last_is_not_zero = False
                start = idx
            else:
                last_is_not_zero = True
        if node != 0 and route_length-start >= min_node_count:  # handle final routes
            map_action_index.append((index, start, route_length))
    map_action_index = np.array(map_action_index, dtype=np.int32)
    return map_action_index


@nb.njit(nogil=True, parallel=True, cache=True)
def _compose_subtsp_coordinates(
    actions: np.ndarray, map_action_index: np.ndarray, coordinates: np.ndarray
):
    n_subtsp = map_action_index.shape[0]
    batch_size = coordinates.shape[0]
    n_samples = actions.shape[0] // batch_size
    max_subtsp_length = (map_action_index[:, 2] - map_action_index[:, 1]).max()
    subtsp_index = np.zeros((n_subtsp, max_subtsp_length + 1), dtype=np.int32)
    subtsp_coordinates = np.zeros(
        (n_subtsp, max_subtsp_length + 1, 2), dtype=coordinates.dtype
    )
    for idx in nb.prange(n_subtsp):
        route_idx, start, end = map_action_index[idx]
        inst_idx = route_idx // n_samples
        subtsp_index[idx, : end - start] = actions[route_idx, start:end]
        subtsp_coordinates[idx, :, :] = coordinates[inst_idx, subtsp_index[idx]]
    return subtsp_index, subtsp_coordinates


@nb.njit(nogil=True, parallel=True, cache=True)
def _update_cvrp_actions(
    cvrp_actions: np.ndarray,
    subtsp_actions: np.ndarray,
    map_action_index: np.ndarray,
    map_node_index: np.ndarray,
):
    subtsp_length = subtsp_actions.shape[1]
    subtsp_underlying_actions = np.take_along_axis(
        map_node_index, subtsp_actions, axis=-1
    )
    for idx in nb.prange(subtsp_actions.shape[0]):
        route_idx, start, end = map_action_index[idx]
        real_nodes = subtsp_underlying_actions[idx]

        # roll the route to ensure it starts with non-zero nodes
        # e.g. [1,2,0,0,3,4] -> [3,4,1,2,0,0]
        shift_count = 0
        last_node = real_nodes[-1]
        for _node_idx in range(subtsp_length):
            current_node = real_nodes[_node_idx]
            if current_node != 0 and last_node == 0:
                shift_count = _node_idx
                break
            last_node = current_node
        if shift_count != 0:
            real_nodes = np.roll(real_nodes, -shift_count)

        # update cvrp_actions
        cvrp_actions[route_idx, start + 1 : end] = real_nodes[real_nodes != 0]
