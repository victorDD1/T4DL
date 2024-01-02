import tyro
import glob
import os
from typing import Optional
from utils.config import Config
from utils.sweep import SweepParametersConfig
from utils.trainer import TrainerSupervised
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

            trainer = TrainerSupervised(
                cfg,
                model,
                dataloader_train,
                dataloader_test,
            )
            trainer.train()

    if test and model_path:
        # MODEL
        model = get_model(model_path)
        model = model.eval()
        # Do what you want

if __name__ == "__main__":
    args = tyro.cli(main)
