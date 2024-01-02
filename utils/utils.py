import glob, os
from torch.nn import Module
from torch import load

from utils.config import Config

DEFAULT_MODEL_PATH = "./models"
# Import all models from files in <DEFAULT_MODEL_PATH>
for p in glob.glob(os.path.join(DEFAULT_MODEL_PATH, "*.py")):
    filename = os.path.split(p)[1].replace(".py", "")
    exec(f"from models.{filename} import *")

class ModelLoader:
    """
    Load model fron model configuration file
    """
    def __init__(self, model_name, cfg_model) -> None:
        self.model = None
        self.model_name = model_name
        self.cfg_model = cfg_model

    def get_model(self) -> Module:
        try:
            exec(f"self.model = {self.model_name}(**self.cfg_model)")
            self.model.__setattr__("name", self.model_name)
            n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Model number of trainable parameters:", n_params)
        except Exception as e:
            print(e)
            print("Can't load model", self.model_name)
        
        return self.model
    
def get_model(state_path:str):
    """
    Return model assuming it is save in logs with its config yaml file
    """
    run_dir = os.path.split(state_path)[0]
    config_path = glob.glob(run_dir + "/*.yaml")[0]
    cfg = Config(config_path)
    cfg_model = {**cfg.model()["PARAMS"]}
    m = ModelLoader(cfg.get_value("model_name"), cfg_model)
    model = m.get_model()
    state = load(state_path)
    model.load_state_dict(state["state_dict"])
    print("Model state restored at epoch", state["epoch"])
    return model