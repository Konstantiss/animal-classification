import torchvision.models as models
from torchvision.transforms import transforms
from torch import nn
from train_and_validate_model import train_validate
from dataloader import *
from resnet import ResNet18
import time
import datetime
import pickle
import matplotlib.pyplot as plt

BATCH_SIZE = 16
EPOCHS = 1
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

start_time = time.time()

mean_loss_per_epoch_train, mean_loss_per_epoch_val = train_validate(model, train_dataloader, val_dataset, loss_fn, optimizer, device, EPOCHS)

execution_time = (time.time() - start_time) / 60
date_time = str(datetime.datetime.now())
model_save_filename = RESULTS_DIR + 'resnet-save-' + date_time + '-' + str(EPOCHS) + '.bin'


torch.save(model.state_dict(), model_save_filename)

results = {
    "model": model.__class__.__name__,
    "train_loss": mean_loss_per_epoch_train,
    "val_loss": mean_loss_per_epoch_val,
    "datetime": datetime.datetime.now(),
    "execution_time": execution_time
}

print('Total execution time: {:.4f} minutes', format(execution_time))
print("Mean training loss per epoch:", mean_loss_per_epoch_train)
print("Validation loss per epoch:", mean_loss_per_epoch_val)

results_filename = RESULTS_DIR + 'results-resnet-' + date_time + '-' + str(EPOCHS) + '.pkl'
with open(results_filename, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

PLOT = True

if PLOT:
    plot_filename = './figs/resnet-loss-plot-train-' + date_time + '-' + str(EPOCHS) + '.png'
    plt.figure(figsize=(10, 5))
    plt.title("Resnet training loss per epoch")
    plt.plot(range(1, EPOCHS + 1), mean_loss_per_epoch_train, linestyle='solid', marker='o', label="Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.xlim(1, )
    # plt.ylim(0, 1)
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()

    plot_filename = './figs/resnet-loss-plot-val-' + date_time + '-' + str(EPOCHS) + '.png'
    plt.figure(figsize=(10, 5))
    plt.title("Resnet validation loss per epoch")
    plt.plot(range(1, EPOCHS + 1), mean_loss_per_epoch_val, linestyle='solid', marker='o', label="Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.xlim(1, )
    # plt.ylim(0, 1)
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()
