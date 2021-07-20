# @author : Abhishek R S

import os
import numpy as np
import tensorflow as tf

"""
- [Nvidia Autopilot - End to End Learning for Self-Driving Cars]
  (https://arxiv.org/pdf/1604.07316.pdf)
"""

class AutoPilot:
    def __init__(self, training_pl, data_format="channels_first"):
        self._training = training_pl
        self._data_format = data_format
        self._initializer = tf.contrib.layers.xavier_initializer_conv2d()

    # define autopilot network
    def auto_pilot_net(self, features):
        if self._data_format == "channels_last":
            features = tf.transpose(features, perm=[0, 2, 3, 1])
        # 3 x 66 x 200

        self.conv1 = self._get_conv2d_layer(
            features, 24, [5, 5], [2, 2], padding="valid", name="conv1")
        self.relu1 = self._get_relu_activation(self.conv1, name="relu1")
        # 24 x 31 x 98

        self.conv2 = self._get_conv2d_layer(
            self.relu1, 36, [5, 5], [2, 2], padding="valid", name="conv2")
        self.relu2 = self._get_relu_activation(self.conv2, name="relu2")
        # 36 x 14 x 47

        self.conv3 = self._get_conv2d_layer(
            self.relu2, 48, [5, 5], [2, 2], padding="valid", name="conv3")
        self.relu3 = self._get_relu_activation(self.conv3, name="relu3")
        # 48 x 5 x 22

        self.conv4 = self._get_conv2d_layer(
            self.relu3, 64, [3, 3], [1, 1], padding="valid", name="conv4")
        self.relu4 = self._get_relu_activation(self.conv4, name="relu4")
        # 64 x 3 x 20

        self.conv5 = self._get_conv2d_layer(
            self.relu4, 64, [3, 3], [1, 1], padding="valid", name="conv5")
        self.relu5 = self._get_relu_activation(self.conv5, name="relu5")
        # 64 x 1 x 18

        self.flatten = tf.layers.flatten(self.relu5, name="flatten")
        self.flatten.set_shape([None, 64 * 1 * 18])

        self.dense1 = self._get_dense_layer(self.flatten, 1164, name="dense1")
        self.dense_relu1 = self._get_relu_activation(self.dense1, name="dense_relu1")
        self.dropout1 = self._get_dropout_layer(
            self.dense_relu1, rate=0.2, name="dense_dropout1")

        self.dense2 = self._get_dense_layer(self.dropout1, 100, name="dense2")
        self.dense_relu2 = self._get_relu_activation(self.dense2, name="dense_relu2")
        self.dropout2 = self._get_dropout_layer(
            self.dense_relu2, rate=0.2, name="dense_dropout2")

        self.dense3 = self._get_dense_layer(self.dropout2, 50, name="dense3")
        self.dense_relu3 = self._get_relu_activation(self.dense3, name="dense_relu3")
        self.dropout3 = self._get_dropout_layer(
            self.dense_relu3, rate=0.2, name="dense_dropout3")

        self.dense4 = self._get_dense_layer(self.dropout3, 10, name="dense4")
        self.dense_relu4 = self._get_relu_activation(self.dense4, name="dense_relu4")
        self.dropout4 = self._get_dropout_layer(
            self.dense_relu4, rate=0.2, name="dense_dropout4")

        self.dense5 = self._get_dense_layer(self.dropout4, 1, name="dense5")
        self.logits = tf.atan(self.dense5, name="logits")

    # convolution2d layer
    def _get_conv2d_layer(self, input_layer, num_filters, kernel_size, strides, padding, name="conv"):
        conv_2d_layer = tf.layers.conv2d(inputs=input_layer, filters=num_filters, kernel_size=kernel_size,
            strides=strides, padding=padding, data_format=self._data_format, kernel_initializer=self._initializer, name=name)
        return conv_2d_layer

    # relu activation
    def _get_relu_activation(self, input_layer, name="relu"):
        relu_layer = tf.nn.relu(input_layer, name=name)
        return relu_layer

    # dropout layer
    def _get_dropout_layer(self, input_layer, rate=0.5, name="dropout"):
        dropout_layer = tf.layers.dropout(inputs=input_layer, rate=rate, training=self._training, name=name)
        return dropout_layer

    # dense layer
    def _get_dense_layer(self, input_layer, num_neurons, use_bias=True, name="dense"):
        dense_layer = tf.layers.dense(input_layer, units=num_neurons, use_bias=use_bias, name=name)
        return dense_layer
