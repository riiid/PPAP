import torch.nn as nn
import torchvision.transforms as T


class AdapterWrapperMidas(nn.Module):
    def __init__(self, midas_model, adapter_class, gamma, lora_alpha):
        super().__init__()
        self.midas = midas_model
        self.add_adapter(adapter_class, gamma, lora_alpha)
        self.model_frozen = False
        self.freeze_model(True)
        self.calculate_training_parameter_ratio()
        self.normalize = T.Compose(
            [
                T.Resize((256, 256)),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
        )

    def add_adapter(self, adapter_class, gamma, lora_alpha):
        """
        Add adapter to midas model
        """
        layer_list = [
            self.midas.pretrained.layer1,
            self.midas.pretrained.layer2,
            self.midas.pretrained.layer3,
            self.midas.pretrained.layer4
        ]
        for layer in layer_list:
            for block_seq in layer:
                if isinstance(block_seq, nn.Sequential):
                    for block in block_seq:
                        target_conv = block.conv_pw
                        adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv)
                        setattr(block, "conv_pw", adapter)

        middle_list = [
            "layer1_rn",
            "layer2_rn",
            "layer3_rn",
            "layer4_rn"
        ]
        for layer_conv_name in middle_list:
            conv = getattr(self.midas.scratch, layer_conv_name)
            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=conv)
            setattr(self.midas.scratch, layer_conv_name, adapter)

        refine_net_list = [
            self.midas.scratch.refinenet1,
            self.midas.scratch.refinenet2,
            self.midas.scratch.refinenet3,
            self.midas.scratch.refinenet4,
        ]
        for index, refine in enumerate(refine_net_list):
            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=refine.out_conv)
            setattr(refine, "out_conv", adapter)
            if index != 3:
                adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=refine.resConfUnit1.conv1)
                setattr(refine.resConfUnit1, "conv1", adapter)

                adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=refine.resConfUnit1.conv2)
                setattr(refine.resConfUnit1, "conv2", adapter)

            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=refine.resConfUnit2.conv1)
            setattr(refine.resConfUnit2, "conv1", adapter)
            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=refine.resConfUnit2.conv2)
            setattr(refine.resConfUnit2, "conv2", adapter)

    def calculate_training_parameter_ratio(self):
        def count_parameters(model, grad):
            return sum(p.numel() for p in model.parameters() if p.requires_grad == grad)

        trainable_param_num = count_parameters(self.midas, True)
        other_param_num = count_parameters(self.midas, False)
        print("Non-trainable parameters:", other_param_num)
        print("Trainable parameters:", trainable_param_num)

        ratio = trainable_param_num / other_param_num
        final_ratio = (ratio / (1 - ratio))
        print("Ratio:", final_ratio)

        return final_ratio

    def forward(self, x):
        x = self.normalize((1 + x) / 2)
        return self.midas(x)

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model."""
        # first freeze/ unfreeze all model weights
        if freeze:
            for n, p in self.named_parameters():
                if 'lora_' not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            for n, p in self.named_parameters():
                if 'bias' in n:
                    if "refinenet4.resConfUnit1" not in n:
                        p.requires_grad = True
                if "bn" in n:
                    p.requires_grad = True
        else:
            for n, p in self.named_parameters():
                p.requires_grad = True
        self.model_frozen = freeze
