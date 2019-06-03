class BaseFeatureExtractor(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)


class FeatureExtractorName(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        ...

        output_prediction = ...

        self.feature_extractor = Model(input_image, output_prediction)
        self.feature_extractor.load_weights('TransferLearningModelDefaultPath')

    def normalize(self, image):
        return image / 255.