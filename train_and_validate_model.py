from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch

def train_validate(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, epochs):
    mean_loss_per_epoch_train = []
    mean_loss_per_epoch_val = []
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=1, factor=0.01)

    print("Model: ", model.__class__.__name__)
    model = model.to(device)
    for epoch in range(epochs):

        print("---- Training ---\n")

        model = model.train()

        losses_per_epoch_train = []
        print("Learning rate: ", optimizer.param_groups[0]['lr'])
        with tqdm(train_dataloader, unit="batch", total=len(train_dataloader)) as tepoch:
            for images, target_classes in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                images = images.to(device)
                target_classes = target_classes.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, target_classes)
                losses_per_epoch_train.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
        current_epoch_loss_train = sum(losses_per_epoch_train) / len(losses_per_epoch_train)
        print(f"Mean training loss for epoch {epoch + 1}:",
              current_epoch_loss_train)
        mean_loss_per_epoch_train.append(current_epoch_loss_train)
        scheduler.step(current_epoch_loss_train)

        print("---- Validation ---\n")

        model = model.val()

        losses_per_epoch_val = []
        with tqdm(val_dataloader, unit="batch", total=len(val_dataloader)) as tepoch:
            for waveform, drrs_true, rt60s_true in tepoch:
                epoch.set_description(f"Epoch {epoch + 1}")
                images = images.to(device)
                target_classes = target_classes.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, target_classes)
                losses_per_epoch_val.append(loss.item())
                tepoch.set_postfix(loss=loss.item())
        current_epoch_loss_val = sum(losses_per_epoch_val) / len(losses_per_epoch_val)
        scheduler.step(current_epoch_loss_val)
        print(f"Mean DRR validation loss for epoch {epoch + 1}:",
              current_epoch_loss_val)
        mean_loss_per_epoch_val.append(current_epoch_loss_val)

    return mean_loss_per_epoch_train, mean_loss_per_epoch_val
