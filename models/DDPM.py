import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from typing import Any, Mapping, Optional

class DDPM(nn.Module):

    DEFAULT_BETA_START = 1e-4
    DEFAULT_BETA_END = 0.02
    DEFAULT_BETASCHEDULE = 'squaredcos_cap_v2'
    DEFAULT_TIMESTEPS = 1000
    DEFAULT_CLIPSAMPLE = False
    DEFAULT_TRAINING_SHAPE = [0,0,0]
    DEFAULT_PREDICTION_TYPE = "epsilon" # "sample" or "epsilon"
    DEFAULT_SELF_CONDITIONING = False

    def __init__(self,
                 model:nn.Module,
                 **kwargs) -> None:
        """
        Diffusion Probabilistic Model (DDPM) Constructor.

        Args:
            - model: The epsilon model.
            - **kwargs: Additional optional parameters for DDPM.
        """

        super(DDPM, self).__init__()
        self.model = model

        self.optional_parameters = {
            **{
            "beta_start" : DDPM.DEFAULT_BETA_START,
            "beta_end" : DDPM.DEFAULT_BETA_END,
            "beta_schedule" : DDPM.DEFAULT_BETASCHEDULE,
            "timesteps" : DDPM.DEFAULT_TIMESTEPS,
            "clip_sample" : DDPM.DEFAULT_CLIPSAMPLE,
            "training_shape" : DDPM.DEFAULT_TRAINING_SHAPE,
            "self_conditioning" : DDPM.DEFAULT_SELF_CONDITIONING,
            "prediction_type" : DDPM.DEFAULT_PREDICTION_TYPE,
            },
            **kwargs # kwargs overrides optional if provided
        }
        self._set_parameters()
        self._set_noise_scheduler()

    def _set_parameters(self):
        """
        Set optional parameters as attributes or buffers.
        """
        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in self.optional_parameters.items():
            try:
                v = torch.tensor(v)
            except:
                setattr(self, k, v)
                continue
            self.register_buffer(k, v)

    def _set_noise_scheduler(self):
        """
        Initialize and set the noise scheduler.
        """
        self.noise_scheduler = DDPMScheduler(
            beta_start=self.beta_start.item(),
            beta_end=self.beta_end.item(),
            beta_schedule=self.beta_schedule, # big impact on performance
            num_train_timesteps=self.timesteps.item(),            
            clip_sample=self.clip_sample.item(), # clip output to [-1,1] to improve stability
            prediction_type=self.prediction_type # our network predicts noise (instead of denoised action)
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict, assign)
        self._set_noise_scheduler()

    def cast_timesteps(self,
                       sample:torch.Tensor,
                       timesteps:Optional[int]):
        """
        Cast and broadcast timesteps.

        Args:
            - sample: The input sample.
            - timesteps: The diffusion steps.

        Returns:
            - The cast and broadcasted timesteps.
        """
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        return timesteps
        
    def forward(self,
                sample:torch.Tensor,
                timesteps:Optional[int] = None,
                global_cond:Optional[torch.Tensor] = None):
        """
        Perform forward diffusion process.

        Args:
            - sample: The input sample.
            - timesteps: The diffusion steps.
            - global_cond: The global conditioning tensor.

        Returns:
            - estimated_noise: Estimated noise.
            - noise: Generated noise.
        """
        if (self.training_shape == torch.zeros((3,), device=sample.device)).all():
            self.training_shape = torch.tensor(sample.shape, device=sample.device).long()
            
        if timesteps == None:
            # Sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (sample.shape[0],),
                device=sample.device
            ).long()

        timesteps = self.cast_timesteps(sample, timesteps)

        # Forward diffusion process
        noise = torch.randn(sample.shape, device=sample.device)
        noisy_data_samples = self.noise_scheduler.add_noise(sample, noise, timesteps)
        
        # Diffusion process
        last_estimate = torch.zeros_like(noise, device=sample.device)
        if torch.rand(1) < int(self.self_conditioning) / 2:
            with torch.no_grad():
                self.model.eval()
                last_estimate = self.model(noisy_data_samples, timesteps, global_cond, self_conditioning=last_estimate)
                self.model.train()

        model_out = self.model(noisy_data_samples, timesteps, global_cond, self_conditioning=last_estimate, return_logits=True)

        return model_out, noise
    
    @torch.no_grad()
    def sample(self,
               size:Optional[torch.Size]=None,
               num_inference_steps:int=-1,
               condition:Optional[torch.Tensor]=None,
               return_intermediate_steps:bool=False
               ):
        """
        Generate samples using backward diffusion process.

        Args:
            - size: The size of the generated samples.
            - num_inference_steps: Number of inference steps.
            - condition: The conditioning tensor.
            - return_intermediate_steps: Return all denoising steps [STEP, B, SAMPLE_SHAPE]

        Returns:
            - intermediate_generated_samples: Intermediate generated samples.
        """

        assert (hasattr(self, "training_shape") or size != None),\
            "Please set attribute sample_shape or provide size argument"
        
        device = "cpu" if not(torch.cuda.is_available()) else "cuda"
        device = condition.device if condition != None else device

        # Get sample shape
        if size != None:
            sample_shape = size.tolist()
        elif hasattr(self, "training_shape"):
            sample_shape = self.training_shape.tolist()
        if condition != None:
            sample_shape[0] = condition.shape[0]
        sample_shape = torch.Size(sample_shape)

        # Initial random noise sample
        generated_samples = torch.randn(sample_shape, device=device)
        if num_inference_steps < 0:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps - 1
        self.noise_scheduler.set_timesteps(num_inference_steps, device)

        # Self conditioning
        last_estimate = torch.zeros(sample_shape, device=device)

        # Intermediate diffusion steps
        intermediate_steps = torch.empty((num_inference_steps + 1, *sample_shape))

        # Backward diffusion process
        for i, t in enumerate(self.noise_scheduler.timesteps):

            if return_intermediate_steps:
                intermediate_steps[i] = generated_samples

            timesteps = self.cast_timesteps(generated_samples, t)
            noise_pred = self.model(generated_samples, timesteps, condition, self_conditioning=last_estimate)

            if self.self_conditioning:
                last_estimate = noise_pred
            
            # Inverse diffusion step (remove noise)
            generated_samples = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=generated_samples
            ).prev_sample

        if return_intermediate_steps:
            intermediate_steps[-1] = generated_samples
            return intermediate_steps
        
        return generated_samples