import yaml
import os
from typing import List

class Config():
    def __init__(self, config_path) -> None:
        self.config_path = config_path
        self.data_cfg = self._load_parameters_from_yaml()

        self.model_keys = "model"
        self.data_keys = "data"
        self.training_keys = "training"
        self.sweep_keys = "sweep"

    def _load_parameters_from_yaml(self):
        with open(self.config_path, 'r') as file:
            parameters = yaml.safe_load(file)
        return parameters
    
    def get_value(self, parameter_str:str, d:dict={}):
        """
        Return parameter value from parameter name
        """
        current_dict = self.data_cfg if len(d) == 0 else d
        for key, value in current_dict.items():
            if isinstance(key, str) and key.lower() == parameter_str.lower():
                return value
            if type(value) == dict:
                res = self.get_value(parameter_str, value)
                if res != None:
                    return res
        return None

    def change_value(self, parameter_str:str, new_value, d:dict={}) -> bool:
        """
        Change value of a config parameter.
        """
        current_dict = self.data_cfg if len(d) == 0 else d
        for key, value in current_dict.items():
            if type(value) == dict:
                self.change_value(parameter_str, new_value, value)
            if key == parameter_str:
                current_dict[key] = new_value
                return True
        return False
    
    def _get_subdict_if_exists(self, key_list: List[str]):
        """
        Returns a subdict of config dict with key in <key_list> (to manage typing errors).
        """
        d = {}
        for model_key in key_list:
            if model_key in self.data_cfg:
                d = self.data_cfg[model_key]
        return d
    
    def get_cfg_as_dict(self, d:dict={}):
        """
        Returns config as a flatten dict.
        """
        flatten_d = {}
        current_dict = self.data_cfg if len(d) == 0 else d
        for key, value in current_dict.items():
            if type(value) == dict:
                flatten_d = {**flatten_d, **self.get_cfg_as_dict(value)}
            else:
                flatten_d[key] = value
        return flatten_d
    
    def write(self, path:str):
        """
        Write config data in a yaml file.
        """
        if path[-5:] != ".yaml" and path[-4:] != ".yml":
            path = os.path.join(path, "config.yaml")

        with open(path, 'w') as file:
            yaml.dump(self.data_cfg, file)

    def model(self):
        return self.get_value(self.model_keys)
    
    def data(self):
        return self.get_value(self.data_keys)
    
    def training(self):
        return self.get_value(self.training_keys)
    
    def sweep(self):
        return self.get_value(self.sweep_keys)
