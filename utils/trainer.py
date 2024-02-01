import os, glob
import torch
import math
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.nn import Module

from utils.config import Config
from utils.logger import Logger, TensorBoardLogger
from utils.log_manager import LogManager

DEFAULT_BACTH_SIZE = 32
DEFAULT_OPTIMIZER = "Adam"
DEFAULT_CRITERION = "MSELoss"
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 100
DEFAULT_LOGDIR = "./logs"
DEFAULT_SCHEDULER = {}
DEFAULT_DATASET = ""
DEFAULT_TRAIN_TEST_RATIO = 5
DEFAULT_USE_LOGGER = True
DEFAULT_MODEL_STATE = ""

class Trainer():
    def __init__(self,
                 cfg:Config,
                 model:Module,
                 dataloader_train:DataLoader,
                 dataloader_test:DataLoader=None,
                 **kwargs) -> None:
        
        self.cfg = cfg
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_epoch = 0
        self.logs_loss_dict = {}
        self.metrics_dict = {}
        self.train_loss_history = []
        self.test_loss_history = []
        self.min_test_loss = math.inf

        # Optional arguments
        self.optional_args = {
            "dataset" : DEFAULT_DATASET,
            "batch_size" : DEFAULT_BACTH_SIZE,
            "epochs" : DEFAULT_EPOCHS,
            "optimizer_str" : DEFAULT_OPTIMIZER,
            "lr" : DEFAULT_LR,
            "criterion_str" : DEFAULT_CRITERION,
            "lr_scheduler" : DEFAULT_SCHEDULER,
            "logdir" : DEFAULT_LOGDIR,
            "train_test_ratio" : DEFAULT_TRAIN_TEST_RATIO,
            "use_logger" : DEFAULT_USE_LOGGER,
            "model_state" : DEFAULT_MODEL_STATE,
        }
        
        self._set_config_params()
        self._set_optimizer()
        self._set_lr_scheduler()
        self._set_criterion()

        # Log manager
        self.log_manager = LogManager(self.logdir)
        self.run_dir = self.log_manager.run_dir
        self.model_path = os.path.join(self.run_dir, f"{cfg.get_value('model_name') if cfg.get_value('model_name') != None else 'model'}.pth")

        # Load model
        if self.model_state != "":
            self.load_state(self.model_state)
        else:
            self.copy_config(cfg.config_path)
                
        # Logger
        self.logger = Logger(self.run_dir)
        if self.use_logger:
            self.logger = TensorBoardLogger(self.run_dir, kwargs.get("comment", ""))

    def _set_config_params(self) -> None:
        """
        Update default params with config.
        """
        for k, v in self.optional_args.copy().items():
            cfg_value = self.cfg.get_value(k)
            if cfg_value != None:
                self.optional_args[k] = cfg_value
                v = cfg_value
            setattr(self, k, v)
        
    def _set_optimizer(self):
        """
        Return torch optimizer from string name.
        """
        try:
            exec(f"setattr(self, 'optimizer', torch.optim.{self.optimizer_str}(self.model.parameters(), lr=self.lr))")
        except AttributeError as e:
            print(e)
            print(f"Optimizer '{self.optimizer_str}' not found.")

    def _set_lr_scheduler(self):
        """
        Return lr_scheduler from string name.
        """
        self.scheduler = None
        if type(self.lr_scheduler) == dict and len(self.lr_scheduler) > 0:
            try:
                lr_scheduler_str = self.lr_scheduler["lr_scheduler_name"]
                lr_scheduler_params = self.lr_scheduler["PARAMS"]
                exec(f"setattr(self, 'scheduler', torch.optim.lr_scheduler.{lr_scheduler_str}(self.optimizer, **lr_scheduler_params))")
            except ValueError as e:
                print(e)
                print(f"Lr scheduler '{self.lr_scheduler}' not found.")
    
    def _set_criterion(self):
        """
        Return criterion from string name.
        """
        try:
            exec(f"setattr(self, 'criterion', torch.nn.{self.criterion_str}())")
        except ValueError as e:
            print(e)
            print(f"Loss '{self.criterion_str}' not found.")

    def _save_model(self) -> None:
        """
        Save model and optim state to run_dir.
        """
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss_history': self.train_loss_history,
            'test_loss_history': self.test_loss_history
        }
        torch.save(state, self.model_path)

    def load_state(self, state_path:str) -> None:
        """
        Load state of pretrained model.
        """
        # If path is run dir
        if state_path[-4:] != ".pth" and self.log_manager._is_run_dir(state_path):
            pth_paths = glob.glob(os.path.split(state_path)[0] + "/*.pth")
            if len(pth_paths) > 0:
                state_path = pth_paths[0]
            else:
                print(f"No .pth in directory {state_path}.")
                return None
        try:
            state = torch.load(state_path)
            self.current_epoch = state["epoch"]
            self.model.load_state_dict(state["state_dict"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.train_loss_history = state["train_loss_history"]
            self.test_loss_history = state["test_loss_history"]

            print(f"Resuming training at {self.run_dir}")

        except Exception as e:
            print(e)
            print(f"Can't load state at {state_path}.")


    def copy_config(self, config_path):
        file_name = os.path.split(config_path)[-1]
        path = os.path.join(self.log_manager.run_dir, file_name)
        self.cfg.write(path)
        
    def write_results(self):
        with open(os.path.join(self.run_dir, "results.txt"), "w") as file:
            file.write(f"Train loss: {self.train_loss_history[-1]}\n")
            file.write(f"Test loss: {self.test_loss_history[-1]}")

    def save_training_curves(self):
        fig, ax = plt.subplots()
        ax.plot(self.train_loss_history, label = "train")
        if self.test_loss_history[-1] != None:
            ax.plot(self.test_loss_history, label = "test")
        plt.yscale("log")
        plt.legend()
        plt.title("Loss history")
        plt.savefig(os.path.join(self.run_dir, "loss_history.png"))

    def write_loss(self):
        self.logger.write_scalars("Loss", self.logs_loss_dict, self.current_epoch)

    def write_hparams(self, cfg:dict):
        self.logger.write_hparams(cfg, self.metrics_dict)

class TrainerSupervised(Trainer):
    def __init__(self,
                 cfg:Config,
                 model:Module,
                 dataloader_train:DataLoader,
                 dataloader_test:DataLoader=None,
                 **kwargs) -> None:
        
        super().__init__(cfg,
                         model,
                         dataloader_train,
                         dataloader_test,
                         **kwargs)

    def _train_epoch(self):
        total_loss = 0.

        for (input, target) in tqdm(self.dataloader_train, desc = "Batch", leave=False):
            # Move batch to device
            input = input.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(input)
            loss = self.criterion(out, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
             
        # Calculate average loss for the epoch
        average_loss = total_loss / len(self.dataloader_train)        
        
        return average_loss

    @torch.no_grad()
    def _test_epoch(self):
        total_loss = 0.

        self.model.eval()
        for (input, target) in tqdm(self.dataloader_test, desc = "Batch", leave=False):
            # Move batch to device
            input = input.to(self.device)
            target = target.to(self.device)

            out = self.model(input)
            loss = self.criterion(out, target)
            total_loss += loss.item()

        # Calculate average loss for the epoch
        average_loss = total_loss / len(self.dataloader_test)

        return average_loss

    def train(self):
        progress_bar = trange(self.epochs, desc = "Epochs")

        for epoch in progress_bar:

            # Train
            train_loss = self._train_epoch()

            # Test
            test_loss = None
            if (self.dataloader_test != None and epoch % self.train_test_ratio == 0):
                test_loss = self._test_epoch()
                # Save model with best test loss
                if (epoch != 0 and test_loss < self.min_test_loss):
                    self._save_model()
                    self.min_test_loss = test_loss
    
            # Write logs and history
                self.logs_loss_dict["tst_loss"] = test_loss

            self.train_loss_history.append(train_loss)
            self.test_loss_history.append(test_loss)

            self.logs_loss_dict["tr_loss"] = train_loss

            self.logger.write_scalars("Loss", self.logs_loss_dict, self.current_epoch)
            progress_bar.set_postfix(self.logs_loss_dict)

            self.current_epoch += 1
            if self.scheduler != None:
                self.scheduler.step()
        
        # Save last epoch model if only train
        if self.dataloader_test == None:
            self._save_model()
            self.metrics_dict["test_loss"] = torch.min(torch.tensor(self.train_loss_history)).item()
        else:
            self.metrics_dict["test_loss"] = self.min_test_loss
        self.write_hparams(self.cfg.get_cfg_as_dict())
        self.write_results()
        self.save_training_curves()
