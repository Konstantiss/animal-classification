from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch

BATCH_SIZE = 16


def test(model, test_dataloader, loss_fn, optimizer, device, epochs):
    true_classes = []
    predicted_classes = []
    predicted_probabilities = []
    mean_loss_per_epoch_test = []
    mean_accuracy_per_epoch_test = []
    all_feature_maps = []

    print("Model: ", model.__class__.__name__)
    model = model.to(device)

    print("---- Testing ---\n")

    losses_per_epoch_test = []
    accuracies_per_epoch_test = []
    print("Learning rate: ", optimizer.param_groups[0]['lr'])
    with tqdm(test_dataloader, unit="batch", total=len(test_dataloader)) as tepoch:
        for images, target_classes in tepoch:
            tepoch.set_description(f"Epoch {1}")
            images = images.to(device)
            target_classes = target_classes.to(device)
            outputs, feature_maps = model(images)
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            true_classes.extend(target_classes.tolist())
            predicted_classes.extend(predictions.tolist())
            predicted_probabilities.extend(outputs.tolist())
            feature_maps = feature_maps.cpu()
            all_feature_maps.extend(feature_maps.tolist())
            correct = (predictions == target_classes).sum().item()
            accuracy = correct / BATCH_SIZE
            accuracies_per_epoch_test.append(accuracy)
            loss = loss_fn(outputs, target_classes)
            losses_per_epoch_test.append(loss.item())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            tepoch.set_postfix(loss=loss.item(), accuracy=accuracy)
            del feature_maps
            torch.cuda.empty_cache()

    current_epoch_loss_test = sum(losses_per_epoch_test) / len(losses_per_epoch_test)
    current_epoch_accuracy_test = sum(accuracies_per_epoch_test) / len(accuracies_per_epoch_test)
    print(f"Mean training loss for epoch {1}:",
          current_epoch_loss_test)
    print(f"Mean training accuracy for epoch {1}:",
          current_epoch_accuracy_test)
    mean_loss_per_epoch_test.append(current_epoch_loss_test)
    mean_accuracy_per_epoch_test.append(current_epoch_accuracy_test)

    return mean_loss_per_epoch_test, mean_accuracy_per_epoch_test, true_classes, predicted_classes, predicted_probabilities, all_feature_maps
