import tyro
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from typing import Optional
from utils.config import Config
from utils.sweep import SweepParametersConfig
from utils.trainer import TrainerDDPM
from utils.utils import get_model, ModelLoader
from data.dataset import get_dataloaders

def main(
    train: bool = False,
    test: bool = False,
    cfg: str = "./config/default_config.yml",
    model_path: Optional[str] = ""
    ) -> None:
    """
    Args:
        train: train model
        test: only test
        cfg: configuration file path (.yml)
        model_path: model path to load (.pth). Load corresponding config.
    """
    ### CONFIG
    if cfg != "":
        cfg = Config(cfg)

    # Load config from config file in run directory
    if model_path != "":
        rundir = os.path.split(model_path)[0]
        cfg_run = glob.glob(rundir + "/*.yaml") + glob.glob(rundir + "/*.yml")
        if len(cfg_run) > 0:
            cfg = Config(cfg_run[0])
        cfg.change_value("logdir", rundir)

    ### DATA
    dataloader_train, dataloader_test = get_dataloaders(**cfg.data())
    
    ### TRAIN
    if train:
        # SWEEP CONFIG
        sweeper = SweepParametersConfig(cfg, cfg.data_cfg.pop("SWEEP", {}))
        for cfg in sweeper:
            # MODEL
            m = ModelLoader(cfg.model()["model_name"], cfg.model()['PARAMS'])
            model = m.get_model()

            trainer = TrainerDDPM(
                cfg,
                model,
                dataloader_train,
            )
            trainer.train()

    if test and model_path:
        INTER_STEPS = 200
        INF_STEPS = 500

        # Sample points
        model = get_model(model_path)
        model = model.eval()

        condition = next(iter(dataloader_train))["condition"]
        generated_samples = model.sample(num_inference_steps=INF_STEPS, condition=condition, intermediate_steps=INTER_STEPS)

        # Plot 
        # Create an animation of intermediate steps
        color = condition.expand(-1, generated_samples.shape[-2]).reshape(-1).detach().numpy()
        x, y = torch.split(generated_samples.reshape(INTER_STEPS, -1, 2), 1, dim=-1)
        x, y = x.squeeze(), y.squeeze()

        fig, ax = plt.subplots(1, figsize=(4, 4))

        LIM = 1.5
        def update(frame):
            ax.clear()
            ax.scatter(x[frame], y[frame], c=color)
            ax.set_aspect('equal', 'box')
            ax.set_xlim([-LIM, LIM])
            ax.set_ylim([-LIM, LIM])

        animation = FuncAnimation(fig, update, frames=INTER_STEPS, repeat=False)
        # Save the animation as a GIF
        animation.save(cfg.get_value("logdir") + "/generated.gif", fps=int(INTER_STEPS / 2))

if __name__ == "__main__":
    args = tyro.cli(main)
