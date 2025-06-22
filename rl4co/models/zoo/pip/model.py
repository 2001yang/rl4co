from typing import Any, Callable
import torch
import torch.nn as nn

from rl4co.data.transforms import StateAugmentation
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.ops import gather_by_index, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class PIP(REINFORCE):
    """PIP Model for handling complex constraints in neural combinatorial optimization based on REINFORCE
    This implements POMO+PIP as described in Bi et al. (2024) https://arxiv.org/pdf/2410.21066.

    Note:
        PIP is a generic and effective framework to advance the capabilities of neural methods towards more complex VRPs.
        First, it integrates the Lagrangian multiplier as a basis to enhance constraint awareness
        and introduces preventative infeasibility masking to proactively steer the solution construction process.
        Moreover, we present PIP-D, which employs an auxiliary decoder and two adaptive strategies
        to learn and predict these tailored masks, potentially enhancing performance
        while significantly reducing computational costs during training.

    Args:
        env: TorchRL Environment
        policy: Policy to use for the algorithm
        policy_kwargs: Keyword arguments for policy
        baseline: Baseline to use for the algorithm. Note that POMO only supports shared baseline,
            so we will throw an error if anything else is passed.
        num_augment: Number of augmentations (used only for validation and test)
        augment_fn: Function to use for augmentation, defaulting to dihedral8
        first_aug_identity: Whether to include the identity augmentation in the first position
        feats: List of features to augment
        num_samples: Number of samples for multi-sampling. If None, use the number of available actions
        **kwargs: Keyword arguments passed to the superclass
    """

    def __init__(
            self,
            env: RL4COEnvBase,
            policy: nn.Module = None,
            policy_kwargs={},
            baseline: str = "shared",
            num_augment: int = 8,
            augment_fn: str | Callable = "dihedral8",
            first_aug_identity: bool = True,
            feats: list = None,
            num_samples: int = None,
            **kwargs,
    ):
        self.save_hyperparameters(logger=False)

        if policy is None:
            policy_kwargs_with_defaults = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            policy_kwargs_with_defaults.update(policy_kwargs)
            policy = AttentionModelPolicy(
                env_name=env.name, **policy_kwargs_with_defaults
            )

        assert baseline == "shared", "POMO only supports shared baseline"

        # Initialize with the shared baseline
        super(PIP, self).__init__(env, policy, baseline, **kwargs)

        self.num_samples = num_samples
        self.num_augment = num_augment
        if self.num_augment > 1:
            self.augment = StateAugmentation(
                num_augment=self.num_augment,
                augment_fn=augment_fn,
                first_aug_identity=first_aug_identity,
                feats=feats,
            )
        else:
            self.augment = None

        # # Add `_multistart` to decode type for train, val and test in policy
        # for phase in ["train", "val", "test"]:
        #     self.set_decode_type_multistart(phase)

    def shared_step(
            self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)
        n_aug, n_sample = self.num_augment, self.num_samples

        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
        elif n_aug > 1:
            td = self.augment(td)

        # Evaluate policy
        out = self.policy(td, self.env, phase=phase, num_samples=n_sample)

        # Unbatchify reward to [batch_size, num_augment, num_samples].
        reward = unbatchify(out["reward"], (n_aug, n_sample))

        # Training phase
        if phase == "train":
            assert n_sample > 1, "num_samples must be > 1 during training"
            log_likelihood = unbatchify(out["log_likelihood"], (n_aug, n_sample))
            relaxed_reward = reward["negative_length"] - reward["total_constraint_violation"] - reward["violated_node_count"]
            out.update({"reward": relaxed_reward})
            self.calculate_loss(td, batch, out, relaxed_reward, log_likelihood)
            penalty = reward["total_constraint_violation"] + reward["violated_node_count"]
            feasible_reward = torch.where(penalty < 1e-6, reward["negative_length"], float('-inf'))
            max_reward, _ = feasible_reward.max(dim=1)
            out.update({"feasible_length": feasible_reward})
            out.update({"max_feasible_length": max_reward})
        # Get multi-sample (=POMO+PIP) rewards and best actions only during validation and test
        else:
            # todo: do not use multi-sample during inference (workaround for now: we use the first solution)
            penalty = reward["total_constraint_violation"][:, :, 0] + reward["violated_node_count"][:, :, 0]
            feasible_reward = torch.where(penalty < 1e-6, reward["negative_length"][:, :, 0], float('-inf'))
            out.update({"reward": feasible_reward})
            out.update({"log_likelihood": unbatchify(out["log_likelihood"], (n_aug, n_sample))[:, :, 0]})
            out.update({"actions": unbatchify(out["actions"], (n_aug, n_sample))[:, :, 0]})
            # Get augmentation score only during inference
            if n_aug > 1:
                # If multisample is enabled, we use the best multisample rewards
                max_aug_reward, max_idxs = feasible_reward.max(dim=1)
                out.update({"max_aug_reward": max_aug_reward})

                if out.get("actions", None) is not None:
                    out.update({"best_aug_actions": gather_by_index(out["actions"], max_idxs[:, None])})

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
