import pickle
import matplotlib.pyplot as plt

pickle_file_path = ''

pkl_contents = []
with (open(pickle_file_path, "rb")) as openfile:
    while True:
        try:
            pkl_contents.append(pickle.load(openfile))
        except EOFError:
            break

mean_loss_per_epoch_train = pkl_contents[0]['train_loss']
mean_accuracy_per_epoch_train = pkl_contents[0]['train_accuracy']
mean_loss_per_epoch_val = pkl_contents[0]['val_loss']
mean_accuracy_per_epoch_val = pkl_contents[0]['val_accuracy']

num_epochs = len(mean_accuracy_per_epoch_train)
# model_name = 'cnn' if pkl_contents[0]['model'] == 'CNNNetwork' else 'resnet'
model_name = pkl_contents[0]['model']
date_time = pkl_contents[0]['datetime']

print('Model:', pkl_contents[0]['model'])
print('Number of epochs:', num_epochs)
print("Mean training loss per epoch:", mean_loss_per_epoch_train)
print("Mean training accuracy per epoch:", mean_accuracy_per_epoch_train)
print("Validation loss per epoch:", mean_loss_per_epoch_val)
print("Validation accuracy per epoch:", mean_accuracy_per_epoch_val)

PLOT = True

if PLOT:
    plot_filename = './figs/resnet-loss-plot-' + str(date_time) + '-' + str(num_epochs) + '.png'
    plt.figure(figsize=(10, 5))
    plt.title("Resnet training and validation loss per epoch")
    plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch_train, linestyle='solid', marker='o', color='r',
             label="Training")
    plt.plot(range(1, num_epochs + 1), mean_loss_per_epoch_val, linestyle='solid', marker='x', color='b',
             label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()

    plot_filename = './figs/resnet-accuracy-plot-' + str(date_time) + '-' + str(num_epochs) + '.png'
    plt.figure(figsize=(10, 5))
    plt.title("Resnet training and validation accuracy per epoch")
    plt.plot(range(1, num_epochs + 1), mean_accuracy_per_epoch_train, linestyle='solid', marker='o', color='r',
             label="Training")
    plt.plot(range(1, num_epochs + 1), mean_accuracy_per_epoch_val, linestyle='solid', marker='x', color='b',
             label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(plot_filename)
    plt.show()