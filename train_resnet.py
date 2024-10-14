import torchvision.models as models
from torchvision.transforms import transforms
from torch import nn
from train_and_validate_model import train_validate
from dataloader import *
from resnet import *
import time
import datetime
import pickle
import matplotlib.pyplot as plt

BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 10 ** -5

training_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] < 3 else x),  # convert images to 3 channels
    transforms.Lambda(lambda x: x[:3] if x.shape[0] > 3 else x),  # convert images to 3 channels
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
])

validation_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] < 3 else x),  # convert images to 3 channels
    transforms.Lambda(lambda x: x[:3] if x.shape[0] > 3 else x),  # convert images to 3 channels
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.4),
    transforms.RandomAutocontrast()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Pytorch running on:", device)

RESULTS_DIR = './results/'

annotations_file_path_train = './train.csv'
annotations_file_path_test = './test.csv'
annotations_file_path_val = './val.csv'

train_dataset = AnimalsDataset(annotations_file_path_train, device, image_transformation=training_transforms)
train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

val_dataset = AnimalsDataset(annotations_file_path_val, device, image_transformation=validation_transforms)
val_dataloader = DataLoader(val_dataset, BATCH_SIZE, shuffle=True)

model = ResNet18().to(device)

loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, momentum=0.8)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

start_time = time.time()

mean_loss_per_epoch_train, mean_accuracy_per_epoch_train, mean_loss_per_epoch_val, mean_accuracy_per_epoch_val = train_validate(
    model, train_dataloader, val_dataloader, loss_fn, optimizer, device, EPOCHS)

execution_time = (time.time() - start_time) / 60
date_time = str(datetime.datetime.now())
model_save_filename = RESULTS_DIR + 'resnet-save-' + date_time + '-' + str(EPOCHS) + '.bin'

torch.save(model.state_dict(), model_save_filename)

results = {
    "model": model.__class__.__name__,
    "train_loss": mean_loss_per_epoch_train,
    "train_accuracy": mean_accuracy_per_epoch_train,
    "val_loss": mean_loss_per_epoch_val,
    "val_accuracy":mean_accuracy_per_epoch_val,
    "datetime": datetime.datetime.now(),
    "execution_time": execution_time
}

print('Total execution time: {:.4f} minutes', format(execution_time))
print("Mean training loss per epoch:", mean_loss_per_epoch_train)
print("Mean training accuracy per epoch:", mean_accuracy_per_epoch_train)
print("Validation loss per epoch:", mean_loss_per_epoch_val)
print("Validation accuracy per epoch:", mean_accuracy_per_epoch_val)

results_filename = './results-resnet-' + date_time + '-' + str(EPOCHS) + '.pkl'
with open(results_filename, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

PLOT = True

if PLOT:
    plot_filename = './figs/resnet-loss-plot-' + date_time + '-' + str(EPOCHS) + '.png'
    plt.figure(figsize=(10, 5))
    plt.title("Resnet training and validation loss per epoch")
    plt.plot(range(1, EPOCHS + 1), mean_loss_per_epoch_train, linestyle='solid', marker='o', color='r',
             label="Training")
    plt.plot(range(1, EPOCHS + 1), mean_loss_per_epoch_val, linestyle='solid', marker='x', color='b',
             label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()

    plot_filename = './figs/resnet-accuracy-plot-' + date_time + '-' + str(EPOCHS) + '.png'
    plt.figure(figsize=(10, 5))
    plt.title("Resnet training and validation accuracy per epoch")
    plt.plot(range(1, EPOCHS + 1), mean_accuracy_per_epoch_train, linestyle='solid', marker='o', color='r',
             label="Training")
    plt.plot(range(1, EPOCHS + 1), mean_accuracy_per_epoch_val, linestyle='solid', marker='x', color='b',
             label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()