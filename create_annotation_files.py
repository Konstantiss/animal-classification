import os
import pandas as pd

PATH = './animal-dataset/train'

directories = []
filenames = []
paths = []
classes = []

for path, dirs, files in os.walk(PATH):
    for file in files:
        class_name = path.rsplit('/', 1)[1].rsplit('_')[0]
        directories.append(path)
        filenames.append(file)
        paths.append(path + '/' + file)
        classes.append(class_name)

all_filenames_and_classes = pd.DataFrame(
    {'directory': directories, 'filename': filenames, 'path': paths, 'class': classes})

all_filenames_and_classes.to_csv('all_filenames_and_classes.csv')

scoiattolo_all = all_filenames_and_classes[all_filenames_and_classes['class'] == 'scoiattolo']
scoiattolo_all.to_csv('scoiattolo_all.csv')

pecora_all = all_filenames_and_classes[all_filenames_and_classes['class'] == 'pecora']
pecora_all.to_csv('pecora_all.csv')

elefante_all = all_filenames_and_classes[all_filenames_and_classes['class'] == 'elefante']
elefante_all.to_csv('elefante_all.csv')

mucca_all = all_filenames_and_classes[all_filenames_and_classes['class'] == 'mucca']
mucca_all.to_csv('mucca_all.csv')

farfalla_all = all_filenames_and_classes[all_filenames_and_classes['class'] == 'farfalla']
farfalla_all.to_csv('farfalla_all.csv')

#### Train - Test - Val split

scoiattolo_train = scoiattolo_all.sample(frac=0.8)
scoiattolo_test = scoiattolo_all.drop(scoiattolo_train.index)
scoiattolo_val = scoiattolo_train.sample(frac=0.2)
scoiattolo_train = scoiattolo_train.drop(scoiattolo_val.index)
scoiattolo_train.to_csv("scoiattolo_train.csv")
scoiattolo_test.to_csv("scoiattolo_test.csv")
scoiattolo_val.to_csv("scoiattolo_val.csv")

pecora_train = pecora_all.sample(frac=0.8)
pecora_test = pecora_all.drop(pecora_train.index)
pecora_val = pecora_train.sample(frac=0.2)
pecora_train = pecora_train.drop(pecora_val.index)
pecora_train.to_csv("pecora_train.csv")
pecora_test.to_csv("pecora_test.csv")
pecora_val.to_csv("pecora_val.csv")

elefante_train = elefante_all.sample(frac=0.8)
elefante_test = elefante_all.drop(elefante_train.index)
elefante_val = elefante_train.sample(frac=0.2)
elefante_train = elefante_train.drop(elefante_val.index)
elefante_train.to_csv("elefante_train.csv")
elefante_test.to_csv("elefante_test.csv")
elefante_val.to_csv("elefante_val.csv")

mucca_train = mucca_all.sample(frac=0.8)
mucca_test = mucca_all.drop(mucca_train.index)
mucca_val = mucca_train.sample(frac=0.2)
mucca_train = mucca_train.drop(mucca_val.index)
mucca_train.to_csv("mucca_train.csv")
mucca_test.to_csv("mucca_test.csv")
mucca_val.to_csv("mucca_val.csv")

farfalla_train = farfalla_all.sample(frac=0.8)
farfalla_test = farfalla_all.drop(farfalla_train.index)
farfalla_val = farfalla_train.sample(frac=0.2)
farfalla_train = farfalla_train.drop(farfalla_val.index)
farfalla_train.to_csv("farfalla_train.csv")
farfalla_test.to_csv("farfalla_test.csv")
farfalla_val.to_csv("farfalla_val.csv")

train_df = pd.concat([scoiattolo_train, pecora_train, elefante_train, mucca_train, farfalla_train], ignore_index=True)
train_df['class'] = train_df['class'].replace({'scoiattolo':1, 'pecora':2, 'elefante':3, 'mucca':4, 'farfalla':5})
train_df.to_csv("train.csv")

test_df = pd.concat([scoiattolo_test, pecora_test, elefante_test, mucca_test, farfalla_test], ignore_index=True)
test_df['class'] = test_df['class'].replace({'scoiattolo':1, 'pecora':2, 'elefante':3, 'mucca':4, 'farfalla':5})
test_df.to_csv("test.csv")

val_df = pd.concat([scoiattolo_val, pecora_val, elefante_val, mucca_val, farfalla_val], ignore_index=True)
val_df['class'] = val_df['class'].replace({'scoiattolo':1, 'pecora':2, 'elefante':3, 'mucca':4, 'farfalla':5})
val_df.to_csv("val.csv")

