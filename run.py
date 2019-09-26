#  csv processing
import pandas as pd
import numpy as np

# keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

# other imports
from time import time


# method to create a dataset returning features x and labels y
def create_dataset(df):
    features = []
    labels = []
    counter = 0

    for index, row in df.sample(frac=1).iterrows():  # iterate over shuffled dataframe
        feature_array = np.array(row[:-1])  # consider all columns but to last one as features
        label = int(row['Class'])  # last row 'Class' as labels

        # class balance is set to 50:50
        if label == 1:
            features.append(feature_array)
            labels.append(label)
        elif label == 0 and counter <= 492:
            features.append(feature_array)
            labels.append(label)
            counter += 1

    features = np.array(features)
    labels = np.array(labels, dtype=int)

    # shuffle created dataset
    p = np.random.permutation(len(features))
    return features[p], labels[p]


# method for setting up deep learning architecture
def create_model():

    model = Sequential()

    model.add(Dense(128, input_dim=input_dim, activation='tanh'))

    model.add(Dense(1024, activation='tanh'))

    model.add(Dense(2048, activation='tanh'))

    model.add(Dense(2048, activation='tanh'))

    model.add(Dense(1024, activation='tanh'))

    model.add(Dense(256, activation='tanh'))

    model.add(Dense(128, activation='tanh'))

    model.add(Dense(1, activation='sigmoid'))

    # adam as optimizer, mean squared error as loss function
    model.compile(optimizer=Adam(lr=lr),
                  loss='mean_squared_error',
                  metrics=['acc'])
    return model


# method returning a list of all callbacks
def define_callbacks():
    return [TensorBoard(log_dir='logs/{}'.format(time())),
            ModelCheckpoint(filepath="nets/{epoch:02d}-{val_acc:.2f}.h5", monitor='val_acc',
                            verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)]


x, y = create_dataset(pd.read_csv('creditcard.csv'))  # loading features x and labels y

# constants used for training
input_dim = x.shape[1]
num_samples = x.shape[0]
batch_size = 32
epochs = 2000
validation_split = 0.2
lr = 0.00001

classifier = create_model()  # define classifier
classifier.fit(x, y,  # fit classifier
               epochs=epochs,
               batch_size=batch_size,
               validation_split=validation_split,
               shuffle=True,
               callbacks=define_callbacks())


