# @author : Abhishek R S

import os
import cv2
import sys
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from autopilot_model import AutoPilot
from autopilot_utils import read_config_file, init

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
param_config_file_name = os.path.join(os.getcwd(), "autopilot_config.json")

# run inference on test set
def infer(FLAGS):
    model_dir = FLAGS.model_dir + str(FLAGS.num_epochs)

    df_data = pd.read_csv(os.path.join(FLAGS.data_dir, "data.txt"), index_col=0)
    df_data_dict = df_data.to_dict("index")

    outputs_dir = "outputs_" + FLAGS.which_set + "_" + str(FLAGS.which_checkpoint_model)
    init(os.path.join(model_dir, outputs_dir))

    print("Preparing inference data.................")
    images_dir_infer = os.path.join(FLAGS.data_dir, "data_resized_2")
    images_list_infer = np.load(
        os.path.join(os.getcwd(), model_dir, "data_dict.npy")
    ).item()[FLAGS.which_set + "_list"]
    print("Preparing inference data completed.......\n")

    print("Building the model.......................")
    image_shape = [None, 3, 66, 200]

    # create image placeholder to input image for inference
    image_pl = tf.placeholder(
        dtype=tf.float32, shape=image_shape, name="input")

    # create training placeholder to control train and inference phase
    training_pl = tf.placeholder(dtype=bool)
    auto_pilot = AutoPilot(training_pl, FLAGS.data_format)
    auto_pilot.auto_pilot_net(image_pl)
    logits = auto_pilot.logits
    print("Building the model completed.............\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ss = tf.Session(config=tf.ConfigProto(device_count={"GPU": 1}))
    ss.run(tf.global_variables_initializer())

    # load the model parameters
    print("Loading the model parameters.............")
    tf.train.Saver().restore(ss,
        os.path.join(os.getcwd(), model_dir, FLAGS.model_file + "-" + str(FLAGS.which_checkpoint_model))
    )
    print("Loading the model parameters completed...\n")

    print("Inference started........................")
    for img_id in images_list_infer:
        img = cv2.cvtColor(cv2.imread(
            os.path.join(images_dir_infer, img_id)), cv2.COLOR_BGR2RGB)
        img_pre = img.astype(np.float32) / 255.
        img_pre = img_pre.reshape(1, img_pre.shape[0], img_pre.shape[1], img_pre.shape[2])
        img_pre = np.transpose(img_pre, [0, 3, 1, 2])

        ti = time.time()
        theta_pred = ss.run(logits,
            feed_dict={image_pl: img_pre, training_pl: False})
        ti = time.time() - ti

        x_ref = 100
        y_ref = 66

        y_gt = 33
        y_pred = 33

        m_gt = np.tan(df_data_dict[img_id]["steering_angle"] * np.pi / 180.)
        m_pred = np.tan(theta_pred)

        x_gt = int(100 - 33 * m_gt)
        x_pred = int(100 - 33 * m_pred)

        img = cv2.line(img, (x_ref, y_ref), (x_gt, y_gt), (0, 255, 0), 2)
        img = cv2.line(img, (x_ref, y_ref), (x_pred, y_pred), (255, 0, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(
            os.getcwd(), model_dir, outputs_dir, img_id), img)
    print("Inference completed......................\n")

    print(f"Predictions saved in : {os.path.join(model_dir, outputs_dir)}\n")
    ss.close()

def main():
    print("Reading the config file..................")
    config = read_config_file(param_config_file_name)
    print("Reading the config file completed........\n")

    data_dir = config["inference"]["data_dir"]
    which_checkpoint_model = config["inference"]["which_checkpoint_model"]
    which_set = config["inference"]["which_set"]

    data_format = config["model"]["data_format"]

    num_epochs = config["training"]["num_epochs"]

    model_dir = config["checkpoint"]["model_dir"]
    model_file = config["checkpoint"]["model_file"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data_dir", default=data_dir, type=str,
        help="directory containing dir with images and labels file")
    parser.add_argument("--which_checkpoint_model", default=which_checkpoint_model,
        type=str, help="checkpoint model to use to run inference")
    parser.add_argument("--which_set", default=which_set, type=str,
        choices=["train", "valid", "test"], help="data to use to run inference")

    parser.add_argument("--data_format", default=data_format, type=str,
        choices=["channels_first", "channels_last"], help="data format")

    parser.add_argument("--num_epochs", default=num_epochs, type=int,
        help="used for correctly fetching the model directory")

    parser.add_argument("--model_dir", default=model_dir,
        type=str, help="directory to load the model")
    parser.add_argument("--model_file", default=model_file,
        type=str, help="file name to load the model")

    FLAGS, unparsed = parser.parse_known_args()
    infer(FLAGS)

if __name__ == "__main__":
    main()
