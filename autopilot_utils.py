# @author : Abhishek R S

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# read the json file and return the content
def read_config_file(json_file_name):
    # open and read the json file
    config = json.load(open(json_file_name))

    # return the content
    return config

# create the model directory if not present
def init(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# parse function for tensorflow dataset api
def parse_fn(img_name, lbl):
    # read
    img_string = tf.read_file(img_name)

    # decode
    img = tf.image.decode_jpeg(img_string, channels=3)

    # convert data to [0, 1] range as float32
    img = tf.image.convert_image_dtype(img, tf.float32)

    # CHW format
    img = tf.transpose(img, perm=[2, 0, 1])

    # convert angle to radians
    lbl = tf.cast(lbl, tf.float32)
    lbl = lbl * np.pi / 180.

    return img, lbl

# return tf dataset
def get_tf_dataset(images_list, labels, num_epochs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images_list, labels))
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(parse_fn, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(batch_size)

    return dataset

# split into train and test set
def get_train_valid_split(images_list, test_size=0.25, random_state=4):
    train_images_list, valid_images_list = train_test_split(
        images_list, test_size=test_size, random_state=random_state)
    valid_images_list, test_images_list = train_test_split(
        valid_images_list, test_size=1. - test_size, random_state=random_state)

    train_images_list = shuffle(np.array(train_images_list))
    valid_images_list = shuffle(np.array(valid_images_list))
    test_images_list = shuffle(np.array(test_images_list))

    return (train_images_list, valid_images_list, test_images_list)
