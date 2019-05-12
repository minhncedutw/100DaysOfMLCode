from KerasUtils import save_trn_history, load_trn_history

class BaseLearner(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_shape, labels, verbose=True):
        raise NotImplementedError("error message")

    def save_weight(self, weight_path: Union[str, Path]):
        self.model.save_weights(filepath=weight_path)

    def load_weight(self, weight_path: Union[str, Path]):
        self.model.load_weights(weight_path)

    def save_model(self, model_path: Union[str, Path]):
        self.model.save(filepath=model_path)

    def load_model(self, model_path: Union[str, Path]):
        self.model = load_model(filepath=model_path)

    def save_history(self, history_path):
        save_trn_history(history=self.trn_his, saving_path=history_path)

    def load_history(self, history_path):
        self.trn_history = load_trn_history(saving_path=history_path)


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
        self.model = CreateNetworkFunction(input_shape=input_shape, n_classes=self.n_classes)

        if verbose: self.model.summary()

    def train(self, trn_generator, val_generator, n_epochs, loss, optimizer, callbacks, metrics=['accuracy'],
              verbose=1):
        ############################################
        # Make train and validation generators
        ############################################

        ############################################
        # Compile the model
        ############################################
        self.optimizer = optimizer
        self.loss = loss
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=metrics)

        ############################################
        # Start the training process
        ############################################
        self.trn_his = self.model.fit_generator(generator=trn_generator, validation_data=val_generator,
                                                epochs=n_epochs, callbacks=callbacks,
                                                workers=1, verbose=verbose)

        ############################################
        # Compute mAP on the validation set
        ############################################

    def predict(self, x, verbose=1):
        start_time_pred = time.time()

        # Normalize data before feed to PointNet
        # x = self.normalizer(x)

        # Predict labels
        prob = self.model.predict(x=np.expand_dims(a=x, axis=0))

        # Decode probabilities to labels
        prob = np.squeeze(prob)  # remove dimension of batch
        pred_labels = prob.argmax(axis=1)

        stop_time_pred = time.time()
        print('Execution Time: ', stop_time_pred - start_time_pred)

        return pred_labels