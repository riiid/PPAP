from peft.resnet_adapter import AdapterWrapperResNet
from peft.deit_adapter import AdapterWrapperDEIT

def add_adapter_for_classifier(classifier_name, classifier, gamma=8, lora_alpha=8):
    if classifier_name in ["ResNet50", "ResNet152", "ResNet18"]:
        from peft.lora import LoraConv2d
        adapter_class = LoraConv2d
        adapter_classifier = AdapterWrapperResNet(classifier, adapter_class, gamma=gamma, lora_alpha=lora_alpha)
    elif classifier_name in ["DEIT"]:
        from peft.lora import LoRALinear
        adapter_class = LoRALinear
        adapter_classifier = AdapterWrapperDEIT(classifier, adapter_class, gamma=gamma, lora_alpha=lora_alpha)
    else:
        raise ValueError(f"classifier_name is not supported for : {classifier_name}")

    adapter_classifier.freeze_model(True)
    adapter_classifier.calculate_training_parameter_ratio()
    return adapter_classifier
