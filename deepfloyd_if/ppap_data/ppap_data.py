"""
Handling dataset loading of ppap data from deepfloyd-if
"""
import random

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
from torchvision.transforms.functional import hflip
import torch.distributed as dist
import os
from PIL import Image

from guided_diffusion.image_datasets import _list_image_files_recursively


class DeepFloydPPAPData(Dataset):
    def __init__(self, data_dir, random_flip=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.random_flip = random_flip

        # load image paths
        self.I_path = os.path.join(self.data_dir, "I")
        self.II_path = os.path.join(self.data_dir, "II")
        self.I_image_list = _list_image_files_recursively(self.I_path)

    def __getitem__(self, item):
        image_path = self.I_image_list[item]
        with Image.open(image_path) as img:
            pil_image_I = img.convert("RGB")

        file_name = os.path.basename(image_path)
        with Image.open(os.path.join(self.II_path, file_name)) as img:
            pil_image_II = img.convert("RGB")

        if self.random_flip and random.random() < 0.5:
            pil_image_I = hflip(pil_image_I)
            pil_image_II = hflip(pil_image_II)

        image_I = self.transform(pil_image_I)
        image_II = self.transform(pil_image_II)
        return {"I": image_I, "II": image_II}

    def __len__(self):
        return len(self.I_image_list)


def load_data(
    data_dir,
    batch_size,
    num_workers=0,
    random_flip=True,
):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5)
    ])
    dataset = DeepFloydPPAPData(data_dir=data_dir, transform=transform, random_flip=random_flip)
    sampler = DistributedSampler(dataset) if dist.get_world_size() > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(sampler is None),
        sampler=sampler,
    )
    return dataloader, sampler
