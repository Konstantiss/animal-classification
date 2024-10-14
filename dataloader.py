import pandas as pd
import os
import torch
from PIL import Image, ImageFile
import torchvision.transforms._functional_tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib as plt

PLOT_IMAGES = False


class AnimalsDataset(Dataset):

    def __init__(self, annotations_file, device, image_transformation=None):
        data = pd.read_csv(annotations_file)
        self.path_list = data['path'].tolist()
        self.class_list = data['class'].tolist()
        self.device = device
        if image_transformation:
            self.image_transformation = image_transformation

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        target_class = self.class_list[idx]
        image = Image.open(path)
        toTensor = transforms.ToTensor()
        image = toTensor(image)
        image = torchvision.transforms._functional_tensor.convert_image_dtype(image, torch.float32)
        if PLOT_IMAGES:
            pass
        if self.image_transformation:
            image = self.image_transformation(image)
        return image, target_class

