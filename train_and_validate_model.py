from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch

BATCH_SIZE = 16

def train_validate(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, epochs):
    mean_loss_per_epoch_train = []
    mean_accuracy_per_epoch_train = []
    mean_loss_per_epoch_val = []
    mean_accuracy_per_epoch_val = []

    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=1, factor=0.01)

    print("Model: ", model.__class__.__name__)
    model = model.to(device)
    for epoch in range(epochs):

        print("---- Training ---\n")

        losses_per_epoch_train = []
        accuracies_per_epoch_train = []
        print("Learning rate: ", optimizer.param_groups[0]['lr'])
        with tqdm(train_dataloader, unit="batch", total=len(train_dataloader)) as tepoch:
            for images, target_classes in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                images = images.to(device)
                target_classes = target_classes.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                correct = (predictions == target_classes).sum().item()
                accuracy = correct / BATCH_SIZE
                accuracies_per_epoch_train.append(accuracy)
                loss = loss_fn(outputs, target_classes)
                losses_per_epoch_train.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)
        current_epoch_loss_train = sum(losses_per_epoch_train) / len(losses_per_epoch_train)
        current_epoch_accuracy_train = sum(accuracies_per_epoch_train) / len(accuracies_per_epoch_train)
        print(f"Mean training loss for epoch {epoch + 1}:",
              current_epoch_loss_train)
        print(f"Mean training accuracy for epoch {epoch + 1}:",
              current_epoch_accuracy_train)
        mean_loss_per_epoch_train.append(current_epoch_loss_train)
        mean_accuracy_per_epoch_train.append(current_epoch_accuracy_train)
        scheduler.step(current_epoch_loss_train)

        print("---- Validation ---\n")

        losses_per_epoch_val = []
        accuracies_per_epoch_val = []
        with tqdm(val_dataloader, unit="batch", total=len(val_dataloader)) as tepoch:
            for images, target_classes in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                images = images.to(device)
                target_classes = target_classes.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                correct = (predictions == target_classes).sum().item()
                accuracy = correct / BATCH_SIZE
                accuracies_per_epoch_val.append(accuracy)
                loss = loss_fn(outputs, target_classes)
                losses_per_epoch_val.append(loss.item())
                tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)
        current_epoch_loss_val = sum(losses_per_epoch_val) / len(losses_per_epoch_val)
        current_epoch_accuracy_val = sum(accuracies_per_epoch_val) / len(accuracies_per_epoch_val)
        print(f"Mean validation loss for epoch {epoch + 1}:",
              current_epoch_loss_val)
        print(f"Mean validation accuracy for epoch {epoch + 1}:",
              current_epoch_accuracy_val)
        mean_loss_per_epoch_val.append(current_epoch_loss_val)
        mean_accuracy_per_epoch_val.append(current_epoch_accuracy_val)

    return mean_loss_per_epoch_train, mean_accuracy_per_epoch_train, mean_loss_per_epoch_val, mean_accuracy_per_epoch_val
