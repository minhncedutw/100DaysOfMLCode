'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# conv_block = Sequential()
# conv_block.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# conv_block.add(Conv2D(64, (3, 3), activation='relu'))
# conv_block.add(MaxPooling2D(pool_size=(2, 2)))
# conv_block.add(Dropout(0.25))
#
# top_block = Sequential()
# top_block.add(Flatten())
# top_block.add(Dense(128, activation='relu'))
# top_block.add(Dropout(0.5))
# top_block.add(Dense(num_classes, activation='softmax'))
#
# model = Sequential()
# model.add(conv_block)
# model.add(top_block)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

import numpy as np
from MinhNC.test.lr_finder import LRFinder
lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-1, steps_per_epoch=np.ceil(len(x_train)/batch_size), epochs=3)
model.fit(x_train, y_train, callbacks=[lr_finder])
lr_finder.plot_loss()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save model
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)
model.save_weights('./last_weight.h5')

# Save test data
import numpy as np
import pickle
# ids = np.arange(start=0, stop=len(x_test))
ids = np.random.randint(low=0, high=len(x_test), size=1000)
with open('./ids.pkl', 'wb') as f:
    pickle.dump(ids, f)

with open('./x_test_1000.pkl', 'wb') as f:
    pickle.dump(x_test[ids], f)

with open('./y_test_1000.tsv', 'w') as f:
    for label in y_test[ids]:
        f.write(str(label) + '\n')

import scipy.misc
def images_to_sprite(data):
    """
    Creates the sprite image
    :param data: [batch_size, height, weight, n_channel]
    :return data: Sprited image::[height, weight, n_channel]
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)

    data = data.reshape((n, n) + data.shape[1:]).transpose(
        (0, 2, 1, 3) + tuple(range(4, data.ndim + 1))
    )
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data
simg = images_to_sprite(x_test)
scipy.misc.imsave('./MNIST_sprites.png', np.squeeze(simg))

# Visualize feature
with open('./model.json') as f:
    config = f.read()
from keras.models import model_from_json, Model
model = model_from_json(config)
model.load_weights('./last_weight.h5')
new_model = Model(model.inputs, model.layers[-3].output)
new_model.set_weights(model.get_weights())

embs_128 = new_model.predict(x_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=32)
embs_128 = pca.fit_transform(embs_128)
with open('./embs_128D.pkl', 'wb') as f:
    pickle.dump(embs_128, f)
embs_128.tofile('./MNIST_tensor.bytes')