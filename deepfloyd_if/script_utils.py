"""
Handling arguments like guided diffusion code.
"""
from deepfloyd_if.modules import IFStageI
import torch

from peft.midas_adapter import AdapterWrapperMidas


def create_model_and_diffusion(token):
    if token:
        from huggingface_hub import login
        login(token=token)
    if_I = IFStageI('IF-I-L-v1.0', device="cpu")
    diffusion = if_I.get_diffusion(None)
    return if_I, diffusion


def create_guidance_model(task_name):
    if task_name == "depth":
        model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        return midas
    else:
        raise NotImplementedError

def add_adapter_for_guidance_model(task_name, model, lora_alpha, gamma):
    if task_name == "depth":
        from peft.lora import LoraConv2d
        adapter_class = LoraConv2d
        adapter_model = AdapterWrapperMidas(model, adapter_class, gamma=gamma, lora_alpha=lora_alpha)
    else:
        raise NotImplementedError
    return adapter_model
