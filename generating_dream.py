from tensorflow.keras.applications.inception_v3 import *
import tensorflow as tf


class Dream:

    """
    DeepDream is the result of an experiment that aimed to visualize the internal patterns that are learned by
    neural network. Compute its gradient with respect to activations of a specific layer and then modify the
    image to increase the magnitude of such activations to in turn magnify the patterns.
    """

    def __init__(self, octave_scale, mixed_layer_names = None):

        self.octave_scale = octave_scale
        self.base_model = InceptionV3(include_top=False, weights='imagenet')

        if mixed_layer_names is None:
            layer_names = ["mixed3","mixed5"]
        else:
            layer_names = mixed_layer_names

        outputs = [self.base_model.get_layer(layer_name)(self.base_model.input) for layer_name in layer_names]

        self.dreamer = tf.keras.Model(self.base_model.input, outputs)

    def _calculate_loss(self, image):

        image_batch = tf.expand_dims(image, 0)

        activations = self.dreamer(image_batch)

        if len(activations) == 1:

            activations = [activations]

        losses = []
        for activation in activations:
            loss = tf.math.reduce_mean(activation)
            losses.append(loss)

        total_loss = tf.math.reduce_sum(losses)

        return total_loss

