from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf
import numpy as np

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

    @tf.function
    def perform_gradient_ascent(self, image, steps, step_size):
        """
        A method to perform gradient ascent

        image = image + d(Loss) / d(Image) * step_size
        """

        loss = tf.constant(0.0)

        for _ in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(image)
                loss = self._calculate_loss(image)

            gradient = tape.gradient(loss, image)

            gradient = gradient / (tf.math.reduce_std(image) + 1e-3)

            image = image + gradient * step_size
            # Clips tensor values to a specified min and max.
            image = tf.clip_by_value(image, -1, 1)

        return loss, image

    def generate_dream(self, image, steps, step_size):

        image = preprocess_input(image)
        image = tf.convert_to_tensor(image)
        step_size = tf.convert_to_tensor(step_size)
        step_size = tf.constant(step_size)
        steps = tf.constant(steps)

        loss, image = self.perform_gradient_ascent(image, steps, step_size)

        image = 255 * (image + 1.0) / 2
        image = tf.cast(image, tf.uint8)

        return np.array(image)








