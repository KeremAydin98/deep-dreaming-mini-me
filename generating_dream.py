from tensorflow.keras.applications.inception_v3 import *
import tensorflow as tf


class Dream:

    """
    DeepDream is the result of an experiment that aimed to visualize the internal patterns that are learned by
    neural network. Compute its gradient with respect to activations of a specific layer and then modify the
    image to increase the magnitude of such activations to in turn magnify the patterns.
    """

    def __init__(self, octave_scale, which_mixed_layer):

        self.octave_scale = octave_scale
        self.mixed = which_mixed_layer

        self.base_model = InceptionV3(include_top=False, weights='imagenet')

        if self.mixed == 7:
            layer_name = "mixed7"
        elif self.mixed == 8:
            layer_name = "mixed8"
        else:
            layer_name = "mixed9"

        self.model = tf.keras.Model(self.base_model.input, self.base_model.get_layer(layer_name)(self.base_model.input))
