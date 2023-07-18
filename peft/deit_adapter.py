import torch.nn as nn
from collections import OrderedDict


class AdapterWrapperDEIT(nn.Module):
    def __init__(self, deit_model, adapter_class, gamma, lora_alpha):
        super().__init__()
        self.deit = deit_model
        self.add_adapter(adapter_class, gamma, lora_alpha)
        self.model_frozen = False
        self.freeze_model(True)
        from guided_diffusion.script_util import get_image_normalization
        self.normalization = get_image_normalization("DEIT")

    def add_adapter(self, adapter_class, gamma, lora_alpha):
        """
        Add adapter to deit
        :param adapter_class: class for adapter
        """
        # Add adapter input convolution.
        for block in self.deit.classifier.blocks:
            adapter = adapter_class(
                gamma,
                lora_alpha,
                block.attn.qkv,
            )
            setattr(block.attn, "qkv", adapter)
            adapter = adapter_class(
                gamma,
                lora_alpha,
                block.attn.proj
            )
            setattr(block.attn, "proj", adapter)

    def calculate_training_parameter_ratio(self):
        def count_parameters(model, grad):
            return sum(p.numel() for p in model.parameters() if p.requires_grad == grad)

        trainable_param_num = count_parameters(self.deit, True)
        other_param_num = count_parameters(self.deit, False)
        print("Non-trainable parameters:", other_param_num)
        print("Trainable parameters:", trainable_param_num)

        ratio = trainable_param_num / other_param_num
        final_ratio = (ratio / (1 - ratio))
        print("Ratio:", final_ratio)

        return final_ratio


    def forward(self, x):
        x = self.normalization((1 + x) / 2)
        return self.deit(x)


    def freeze_model(self, freeze=True):
        """Freezes all weights of the model."""
        if freeze:
            # First freeze/ unfreeze all model weights
            # first freeze/ unfreeze all model weights
            for n, p in self.named_parameters():
                if 'lora_' not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            for n, p in self.named_parameters():
                if 'bias' in n:
                    p.requires_grad = True
        else:
            # Unfreeze
            for n, p in self.named_parameters():
                p.requires_grad = True
        self.model_frozen = freeze


    def adapter_state_dict(self):
        """
        Save only adapter parts
        """
        state_dict = self.state_dict()
        adapter_dict = OrderedDict()
        for name, param in state_dict.items():
            if "lora_" in name:
                adapter_dict[name] = param
            elif "bias" in name:
                adapter_dict[name] = param
        return adapter_dict


if __name__ == "__main__":
    from peft.lora import LoRALinear
    from guided_diffusion.script_util import create_pretrained_classifier, get_image_normalization
    import torch
    adapter_class = LoRALinear
    deit = create_pretrained_classifier(classifier_name="DEIT")
    adapter_deit = AdapterWrapperDEIT(deit, adapter_class, gamma=8, lora_alpha=8)

    adapter_deit.freeze_model(True)
    adapter_deit.calculate_training_parameter_ratio()

    state_dict = adapter_deit.adapter_state_dict()
    torch.save(state_dict, "/tmp/ss.ckpt")
    adapter_deit.load_state_dict(torch.load("/tmp/ss.ckpt"), strict=False)
