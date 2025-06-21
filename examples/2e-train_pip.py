from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger
import torch
from rl4co.envs import TSPTWEnv
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.trainer import RL4COTrainer

def main():
    # Set device
    device_id = 3

    # RL4CO env based on TorchRL
    env = TSPTWEnv(generator_params={'num_loc': 50, 'hardness': "hard", "pip_step": 1}) # pip_step: -1 (no mask), 0 (mask), 1 (pip mask in Bi, et al.,2024)

    # Policy: neural network, in this case with encoder-decoder architecture
    # Note that this is adapted the same as POMO did in the original paper
    policy = AttentionModelPolicy(env_name=env.name,
                                  embed_dim=128,
                                  num_encoder_layers=6,
                                  num_heads=8,
                                  normalization="instance",
                                  use_graph_context=False
                                  )

    # RL Model (POMO)
    model = POMO(env,
                 policy,
                 batch_size=128,
                 train_data_size=10000*2,
                 val_data_size=1000,
                 optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-6},
                 )

    # Example callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # save to checkpoints/
        filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
        save_top_k=1,  # save only the best model
        save_last=True,  # save the last model
        monitor="val/reward",  # monitor validation reward
        mode="max",
    )  # maximize validation reward
    rich_model_summary = RichModelSummary(max_depth=3)  # model summary callback
    callbacks = [checkpoint_callback, rich_model_summary]

    # Logger
    logger = WandbLogger(project="rl4co", name=f"{env.name}_pip")
    # logger = None # uncomment this line if you don't want logging

    # Main trainer configuration
    trainer = RL4COTrainer(
        max_epochs=1,
        accelerator="gpu",
        devices=device_id,
        logger=logger,
        callbacks=callbacks,
    )

    # Main training loop
    trainer.fit(model)

    # Greedy rollouts over trained model
    # note: modify this to load your own data instead!
    td_init = env.reset(batch_size=[16])
    policy = model.policy
    out = policy(td_init.clone(), env, phase="test", decode_type="greedy")

    # Print results
    # Fixme: also output the penalty
    print(f"Tour lengths: {[f'{-r.item():.3f}' for r in out['reward']]}")
    print(f"Avg tour length: {-torch.mean(out['reward']).item():.3f}")

if __name__ == "__main__":
    main()

