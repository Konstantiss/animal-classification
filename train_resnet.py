import torchvision.models as models
from torchvision.transforms import transforms
from torch import nn
from train_and_validate_model import train_validate
from dataloader import *
from resnet import ResNet18
import time
import matplotlib.pyplot as plt

BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 10**-3

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] < 3 else x),  # convert images to 3 channels
    transforms.Lambda(lambda x: x[:3] if x.shape[0] > 3 else x),  # convert images to 3 channels
    #transforms.RandomHorizontalFlip(p=0.5),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Pytorch running on:", device)


RESULTS_DIR = './results/'

annotations_file_path_train = './train.csv'
annotations_file_path_test = './test.csv'
annotations_file_path_val = './val.csv'

train_dataset = AnimalsDataset(annotations_file_path_train, device, image_transformation=image_transforms)
train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False)

val_dataset = AnimalsDataset(annotations_file_path_val, device, image_transformation=image_transforms)
val_dataloader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

model = ResNet18().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001, momentum=0.9)

mean_loss_per_epoch_train, mean_loss_per_epoch_val = train_validate(model, train_dataloader, val_dataset, loss_fn, optimizer, device, EPOCHS)