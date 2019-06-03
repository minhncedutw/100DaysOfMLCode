from pathlib import Path
from typing import List, Tuple, Union, Any
import argparse

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # The GPU id to use, usually either "0" or "1"

import numpy as np
import json

import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.models import load_model
from KerasUtils import save_trn_history

class BaseLearner(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_shape, labels, verbose=True):
        raise NotImplementedError("error message")

    def save_weight(self, weight_path: Union[str, Path]):
        success = self.model.save_weights(filepath=weight_path)
        print('Save weight {}: {}'.format(weight_path, success))

    def load_weight(self, weight_path: Union[str, Path]):
        success = self.model.load_weights(weight_path)
        self.model._make_predict_function() # safe thread
        self.graph = tf.get_default_graph() # save graph of current model
        self.predict(np.ones(self.input_shape)) # initialize the network in the 1st time
        print('Load weight {}: {}'.format(weight_path, success))

    def save_model(self, model_path: Union[str, Path]):
        success = self.model.save(filepath=model_path)
        print('Save model {}: {}'.format(model_path, success))

    def load_model(self, model_path: Union[str, Path]):
        try:
            self.model = load_model(filepath=model_path)
            self.model._make_predict_function()  # safe thread
            self.graph = tf.get_default_graph()  # save graph of current model
            self.predict(np.ones(self.input_shape))  # initialize the network in the 1st time
            print('Load model {}: successful'.format(model_path))
        except IOError:
            print('An error occured trying to read the file.')

    def save_history(self, history_path):
        save_trn_history(history=self.model.trn_his, saving_path=history_path)


class PointNetLearner(BaseLearner):
    def __init__(self, input_shape, labels, verbose=True):
        """
        Description:
        :param NAME: TYPE, MEAN
        :return: TYPE, MEAN
        """
        '''**************************************************************
        I. Set parameters
        '''
        self.input_shape = input_shape
        self.labels = list(labels)
        self.n_classes = len(self.labels)

        '''**************************************************************
        II. Make the models
        '''
        self.model = create_model(input_shape=input_shape, n_classes=n_classes)
        self.model._make_predict_function()  # have to initialize before threading
        self.graph = None

        if verbose: self.model.summary()

    def train(self, trn_data, tes_data, n_epochs, loss, optimizer=Adam, callbacks=None, metrics=['accuracy'],
              verbose=1):
        ############################################
        # Make train and validation generators
        ############################################

        ############################################
        # Compile the model
        ############################################

        ############################################
        # Start the training process
        ############################################

        ############################################
        # Compute mAP on the validation set
        ############################################

    def predict(self, x, verbose=0):
        # Normalize data before feed to PointNet
        # x = self.normalizer(x)

        # Predict labels
        with self.graph.as_default():
            prob = self.model.predict(x=np.expand_dims(a=x, axis=0))

        # Decode probabilities to labels
        prob = np.squeeze(prob)  # remove dimension of batch
        pred_labels = prob.argmax(axis=1)

        return pred_labels


def _main_(args):
    print('Hello World! This is {:s}'.format(args.desc))

    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())


    ##-----------------DO SMT HERE----------------##


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Your program name!!!')
    argparser.add_argument('-d', '--desc', help='description of the program', default='HANDBOOK')
    argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)