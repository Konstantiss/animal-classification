import torchvision.models as models
from torchvision.transforms import transforms
from torch import nn
import torch
from test_model import test
from dataloader import *
from resnet import *
import time
import numpy as np
import seaborn as sn
import datetime
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 10 ** -5

test_transforms = transforms.Compose([
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

test_dataset = AnimalsDataset(annotations_file_path_test, device, image_transformation=test_transforms)
test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

model = ResNet18().to(device)

model.load_state_dict(torch.load('./results/resnet-save-2024-10-14 20:50:37.461546-15.bin'))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, momentum=0.8)

start_time = time.time()

mean_loss_per_epoch_test, mean_accuracy_per_epoch_test, true_classes, predicted_classes, predicted_probabilities, all_feature_maps = test(
    model, test_dataloader, loss_fn, optimizer, device, EPOCHS)

classes = ['scoiattolo', 'pecora', 'elefante', 'mucca', 'farfalla']
colors = ['Greys', 'Purples', 'Greens', 'Oranges', 'Reds']
for index, (feature_map, true_class) in enumerate(zip(all_feature_maps, true_classes)):

    gray_scale = torch.sum(torch.tensor(feature_map), 0)
    gray_scale = gray_scale / len(feature_map)
    fig = plt.figure()
    imgplot = plt.imshow(gray_scale, cmap=colors[true_class])
    plt.axis("off")
    plt.title(classes[true_class])
    plt.savefig('./feature-maps/' + str(index) + '.png')
    plt.close()
    all_feature_maps.remove(feature_map)

execution_time = (time.time() - start_time) / 60
date_time = str(datetime.datetime.now())

results = {
    "model": model.__class__.__name__,
    "test_loss": mean_loss_per_epoch_test,
    "test_accuracy": mean_accuracy_per_epoch_test,
    "datetime": datetime.datetime.now(),
    "execution_time": execution_time
}

print('Total execution time: {:.4f} minutes', format(execution_time))
print("Mean test loss per epoch:", mean_loss_per_epoch_test)
print("Mean test accuracy per epoch:", mean_accuracy_per_epoch_test)

results_filename = './results-resnet-test' + date_time + '-' + str(EPOCHS) + '.pkl'
with open(results_filename, 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

classes = ['scoiattolo', 'pecora', 'elefante', 'mucca', 'farfalla']

confusion_matrix = confusion_matrix(true_classes, predicted_classes)
dataframe_confusion_matrix = pd.DataFrame(confusion_matrix / np.sum(confusion_matrix, axis=1)[:, None],
                                          index=[i for i in classes],
                                          columns=[i for i in classes])
plt.figure(figsize=(12, 7))
sn.heatmap(dataframe_confusion_matrix, annot=True)
plt.show()
plt.savefig('./figs/confusion-matrix.png')

print("Precision score for each class: [scoiattolo, pecora, elefante, mucca, farfalla]")
print(precision_score(true_classes, predicted_classes, average=None))
print("Precision score (macro):")
print(precision_score(true_classes, predicted_classes, average='macro'))
print("Recall score for each class: [scoiattolo, pecora, elefante, mucca, farfalla]")
print(recall_score(true_classes, predicted_classes, average=None))
print("Recall score (macro):")
print(recall_score(true_classes, predicted_classes, average='macro'))
print("F1 score for each class: [scoiattolo, pecora, elefante, mucca, farfalla]")
print(f1_score(true_classes, predicted_classes, average=None))
print("F1 score (macro):")
print(f1_score(true_classes, predicted_classes, average='macro'))
print("Micro-averaged One-vs-Rest ROC AUC score:")
print(roc_auc_score(true_classes, predicted_probabilities, multi_class="ovr", average="micro"))
print("Macro-averaged One-vs-Rest ROC AUC score:")
print(roc_auc_score(true_classes, predicted_probabilities, multi_class="ovr", average="macro"))
print("One-vs-Rest ROC AUC score for each class:")
print(roc_auc_score(true_classes, predicted_probabilities, multi_class="ovr", average=None))

