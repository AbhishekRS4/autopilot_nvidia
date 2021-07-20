# @author : Abhishek R S

import os
import sys
import time
import math
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from autopilot_model import AutoPilot
from autopilot_utils import init, read_config_file, get_train_valid_split, get_tf_dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
param_config_file_name = os.path.join(os.getcwd(), "autopilot_config.json")

# return mean squared loss
def compute_loss(groundtruth, prediction, name="mean_squared_error"):
    mean_error = tf.reduce_mean(tf.squared_difference(groundtruth, prediction))

    return mean_error

# return the optimizer op which has to be used to minimize the loss function
def get_optimizer_decay(initial_learning_rate, loss_function, global_step, epsilon=0.00001):
    decay_steps = 300
    end_learning_rate = 0.000001
    decay_rate = 0.97
    power = 0.95
    learning_rate = tf.train.polynomial_decay(initial_learning_rate, global_step,
        decay_steps, end_learning_rate, power=power)

    adam_optimizer_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate, epsilon=epsilon).minimize(loss_function)

    return adam_optimizer_op

# save the trained model
def save_model(session, model_dir, model_file, epoch):
    saver = tf.train.Saver()
    saver.save(session, os.path.join(os.getcwd(), model_dir, model_file),
        global_step=(epoch + 1))

# start batch training of the network
def batch_train(FLAGS):
    print("Initializing.............................")
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    model_dir = FLAGS.model_dir + str(num_epochs)
    init(model_dir)
    print("Initializing completed...................\n")

    print("Preparing training meta data.............")
    # read csv file containing image names and labels
    df_data = pd.read_csv(
        os.path.join(FLAGS.data_dir, "data.txt"), usecols=["image_id", "steering_angle"])

    # divide data into train, valid and test
    df_data_train, df_data_valid, df_data_test = get_train_valid_split(df_data)

    # create dictionary with train, valid and test set samples
    dict_data_list = dict()
    dict_data_list["train_list"] = df_data_train[:, 0]
    dict_data_list["valid_list"] = df_data_valid[:, 0]
    dict_data_list["test_list"] = df_data_test[:, 0]

    np.save(os.path.join(model_dir, "data_dict.npy"), np.array(dict_data_list))

    # train and validation image ids
    arr_images_train = np.array(df_data_train[:, 0], dtype=np.str)
    arr_images_valid = np.array(df_data_valid[:, 0], dtype=np.str)

    # train and validation steering angle labels
    arr_labels_train = np.array(df_data_train[:, 1:], dtype=np.float32)
    arr_labels_valid = np.array(df_data_valid[:, 1:], dtype=np.float32)

    # attach path to image ids
    arr_images_train = [os.path.join(
        FLAGS.data_dir, "data_resized_2", x) for x in arr_images_train]
    arr_images_valid = [os.path.join(
        FLAGS.data_dir, "data_resized_2", x) for x in arr_images_valid]

    # convert to array
    arr_images_train = np.array(arr_images_train, dtype=np.str)
    arr_images_valid = np.array(arr_images_valid, dtype=np.str)

    # compute number of train and validation batches
    num_samples_train = arr_images_train.shape[0]
    num_batches_train = int(math.ceil(num_samples_train / float(batch_size)))

    num_samples_valid = arr_images_valid.shape[0]
    num_batches_valid = int(math.ceil(num_samples_valid / float(batch_size)))
    print("Preparing training meta data completed...\n")

    print("Building the model.......................")
    # create train and validation tf datasets
    dataset_train = get_tf_dataset(
        arr_images_train, arr_labels_train, num_epochs, batch_size)
    dataset_valid = get_tf_dataset(
        arr_images_valid, arr_labels_valid, num_epochs, batch_size)

    # create iterator
    iterator = tf.data.Iterator.from_structure(
        dataset_train.output_types, dataset_train.output_shapes)
    features, labels = iterator.get_next()

    # create initializers for train and validation tf dataset
    init_op_train = iterator.make_initializer(dataset_train)
    init_op_valid = iterator.make_initializer(dataset_valid)

    # create training placeholder to control train and inference phase
    training_pl = tf.placeholder(dtype=bool)

    # create global step placeholder
    global_step = tf.placeholder(tf.int32)

    # create autopilot object
    auto_pilot = AutoPilot(training_pl, FLAGS.data_format)
    auto_pilot.auto_pilot_net(features)
    logits = auto_pilot.logits

    # get all variables for applying weight decay
    train_var_list = [v for v in tf.trainable_variables()]

    # compute overall loss
    loss_1 = compute_loss(labels, logits)
    loss_2 = FLAGS.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])
    loss = loss_1 + loss_2

    # get optimizer operation
    optimizer_op = get_optimizer_decay(FLAGS.learning_rate, loss, global_step)
    print("Building the model completed.............\n")

    print(f"Number of epochs to train : {num_epochs}")
    print(f"Batch size : {batch_size}")
    print(f"Number of train samples : {num_samples_train}")
    print(f"Number of train batches : {num_batches_train}")
    print(f"Number of validation samples : {num_samples_valid}")
    print(f"Number of validation batches : {num_batches_valid}\n")

    print("Training the model.......................")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ss = tf.Session(config=tf.ConfigProto(device_count={"GPU": 1}))
    ss.run(tf.global_variables_initializer())

    train_loss_per_epoch = list()
    valid_loss_per_epoch = list()

    for epoch in range(num_epochs):
        ti = time.time()
        temp_train_loss_per_epoch = 0
        temp_valid_loss_per_epoch = 0

        ss.run(init_op_train)
        for _ in range(num_batches_train):
            _, loss_per_batch = ss.run(
                [optimizer_op, loss],
                feed_dict={training_pl: True, global_step: epoch})
            temp_train_loss_per_epoch += loss_per_batch

        ss.run(init_op_valid)
        for _ in range(num_batches_valid):
            loss_per_batch = ss.run(loss, feed_dict={training_pl: False})
            temp_valid_loss_per_epoch += loss_per_batch

        ti = time.time() - ti
        train_loss_per_epoch.append(temp_train_loss_per_epoch)
        valid_loss_per_epoch.append(temp_valid_loss_per_epoch)

        print(f"Epoch : {epoch+1} / {FLAGS.num_epochs}, time taken : {ti:.2f} sec.")
        print(f"training loss : {temp_train_loss_per_epoch/num_batches_train:.4f}, \
            validation loss : {temp_valid_loss_per_epoch/num_batches_valid:.4f}")

        if (epoch + 1) % FLAGS.checkpoint_epoch == 0:
            save_model(ss, model_dir, FLAGS.model_file, epoch)
    print("Training the model completed.............\n")

    print("Saving the model.........................")
    save_model(ss, model_dir, FLAGS.model_file, epoch)

    train_loss_per_epoch = np.array(train_loss_per_epoch)
    valid_loss_per_epoch = np.array(valid_loss_per_epoch)

    train_loss_per_epoch = np.true_divide(train_loss_per_epoch, num_batches_train)
    valid_loss_per_epoch = np.true_divide(valid_loss_per_epoch, num_batches_valid)

    losses_dict = dict()
    losses_dict["train_loss"] = train_loss_per_epoch
    losses_dict["valid_loss"] = valid_loss_per_epoch

    np.save(os.path.join(os.getcwd(), model_dir, FLAGS.model_metrics), np.array(losses_dict))
    print("Saving the model completed...............\n")
    ss.close()

def main():
    print("Reading the config file..................")
    config = read_config_file(param_config_file_name)
    print("Reading the config file completed........\n")

    data_dir = config["data"]["data_dir"]

    data_format = config["model"]["data_format"]

    learning_rate = config["training"]["learning_rate"]
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    weight_decay = config["training"]["weight_decay"]
    checkpoint_epoch = config["training"]["checkpoint_epoch"]

    model_dir = config["checkpoint"]["model_dir"]
    model_file = config["checkpoint"]["model_file"]
    model_metrics = config["checkpoint"]["model_metrics"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", default=data_dir, type=str,
        help="directory containing dir with images and labels file")

    parser.add_argument("--data_format", default=data_format, type=str,
        choices=["channels_first", "channels_last"], help="data format")

    parser.add_argument("--learning_rate", default=learning_rate,
        type=float, help="learning rate")
    parser.add_argument("--num_epochs", default=num_epochs,
        type=int, help="number of epochs to train")
    parser.add_argument("--batch_size", default=batch_size,
        type=int, help="number of samples in a batch")
    parser.add_argument("--weight_decay", default=weight_decay,
        type=float, help="weight decay")
    parser.add_argument("--checkpoint_epoch", default=checkpoint_epoch,
        type=int, help="checkpoint epoch to save every kth model")

    parser.add_argument("--model_dir", default=model_dir,
        type=str, help="directory to save the model")
    parser.add_argument("--model_file", default=model_file,
        type=str, help="file name to save the model")
    parser.add_argument("--model_metrics", default=model_metrics,
        type=str, help="file name to save metrics")

    FLAGS, unparsed = parser.parse_known_args()
    batch_train(FLAGS)

if __name__ == "__main__":
    main()
