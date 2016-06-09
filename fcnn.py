from scipy.misc import imresize

import dicom
import fnmatch
import numpy as np
import os

from initializer import *

# train and validate images will be resized to square with side
IMAGE_SIZE = 256
BATCH_SIZE = 1


def convolution2d(x_img, W_conv):
    return tf.nn.conv2d(x_img, W_conv, strides=[1, 1, 1, 1], padding='SAME')


def deconvolution2d(x_img, W_conv, output_shape, strides=None):
    if strides is None:
        strides = [1, 1, 1, 1]
    return tf.nn.conv2d_transpose(x_img, W_conv, output_shape=output_shape, strides=strides, padding='SAME')


def max_pool_2x2(x_img):
    return tf.nn.max_pool(x_img, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class FCNN:
    def __init__(self, train_dir=None, val_dir=None):
        self.train_dir = train_dir
        self.val_dir = val_dir

        self.train_images = []
        self.train_labels = []

        self.val_images = []
        self.val_labels = []

        # Convolution
        self.W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
        self.b_conv1 = bias_variable([32], "b_conv1")

        self.W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
        self.b_conv2 = bias_variable([64], "b_conv2")

        self.W_conv3 = weight_variable([3, 3, 64, 64], "W_conv3")
        self.b_conv3 = bias_variable([64], "b_conv3")

        self.W_conv4 = weight_variable([3, 3, 64, 5], "W_conv4")
        self.b_conv4 = bias_variable([5], "b_conv4")

        self.x = tf.placeholder(tf.float32, shape=[None, np.square(IMAGE_SIZE)])
        self.y_ = tf.placeholder(tf.float32, shape=[None, np.square(IMAGE_SIZE)])

        self.x_img = tf.reshape(self.x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

        self.h_conv1 = tf.nn.sigmoid(convolution2d(self.x_img, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)
        self.h_conv2 = tf.nn.sigmoid(convolution2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)
        self.h_conv3 = tf.nn.sigmoid(convolution2d(self.h_pool2, self.W_conv3) + self.b_conv3)
        self.h_pool3 = max_pool_2x2(self.h_conv3)
        self.h_conv4 = tf.nn.sigmoid(convolution2d(self.h_pool3, self.W_conv4) + self.b_conv4)

        # Deconvolution
        self.W_deconv4 = weight_variable([3, 3, 64, 5], "W_deconv4")
        self.b_deconv4 = bias_variable([5], "b_deconv4")
        self.deconv4_shape = tf.pack([BATCH_SIZE, 32, 32, 64])

        self.W_pool3 = weight_variable([2, 2, 64, 64], "W_pool3")
        self.b_pool3 = bias_variable([64], "b_pool3")
        self.pool3_shape = tf.pack([BATCH_SIZE, 64, 64, 64])

        self.W_deconv3 = weight_variable([3, 3, 64, 64], "W_deconv3")
        self.b_deconv3 = bias_variable([5], "b_deconv3")
        self.deconv3_shape = tf.pack([BATCH_SIZE, 64, 64, 64])

        self.W_pool2 = weight_variable([2, 2, 64, 64], "W_pool2")
        self.b_pool2 = bias_variable([64], "b_pool2")
        self.pool2_shape = tf.pack([BATCH_SIZE, 128, 128, 64])

        self.W_deconv2 = weight_variable([5, 5, 32, 64], "W_deconv2")
        self.b_deconv2 = bias_variable([32], "b_deconv2")
        self.deconv2_shape = tf.pack([BATCH_SIZE, 128, 128, 32])

        self.W_pool1 = weight_variable([2, 2, 32, 32], "W_pool1")
        self.b_pool1 = bias_variable([32], "b_pool1")
        self.pool1_shape = tf.pack([BATCH_SIZE, 256, 256, 32])

        self.W_deconv1 = weight_variable([5, 5, 1, 32], "W_deconv1")
        self.b_deconv1 = bias_variable([1], "b_deconv1")
        self.deconv1_shape = tf.pack([BATCH_SIZE, 256, 256, 1])

        self.h_deconv4 = tf.nn.sigmoid(deconvolution2d(self.h_conv4, self.W_deconv4, self.deconv4_shape))
        self.h_depool3 = tf.nn.sigmoid(deconvolution2d(self.h_deconv4, self.W_pool3, self.pool3_shape, [1, 2, 2, 1]))
        self.h_deconv3 = tf.nn.sigmoid(deconvolution2d(self.h_depool3, self.W_deconv3, self.deconv3_shape))
        self.h_depool2 = tf.nn.sigmoid(deconvolution2d(self.h_deconv3, self.W_pool2, self.pool2_shape, [1, 2, 2, 1]))
        self.h_deconv2 = tf.nn.sigmoid(deconvolution2d(self.h_depool2, self.W_deconv2, self.deconv2_shape))
        self.h_depool1 = tf.nn.sigmoid(deconvolution2d(self.h_deconv2, self.W_pool1, self.pool1_shape, [1, 2, 2, 1]))
        self.h_deconv1 = tf.nn.sigmoid(deconvolution2d(self.h_depool1, self.W_deconv1, self.deconv1_shape))

        self.y = tf.reshape(self.h_deconv1, [-1, np.square(IMAGE_SIZE)])
        self.error = tf.nn.l2_loss(self.y_ - self.y)
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.error)

        self.session_started = False
        self.session = tf.Session()

        self.save_path = None
        self.saver = tf.train.Saver()

    def start_session(self):
        init = tf.initialize_all_variables()
        self.session.run(init)
        self.session_started = True

    def set_train_dir(self, train_dir):
        self.train_dir = train_dir
        self.load_train_data()

    def set_val_dir(self, val_dir):
        self.val_dir = val_dir
        self.load_val_data()

    def load_train_data(self):
        if self.train_dir:
            dcm_images = sorted([os.path.join(path, f)
                                 for path, _, files in os.walk(self.train_dir)
                                 for f in fnmatch.filter(files, 'image*.dcm')])

            dcm_labels = sorted([os.path.join(path, f)
                                 for path, _, files in os.walk(self.train_dir)
                                 for f in fnmatch.filter(files, 'ribsMask*.dcm')])

            for img, lbl in zip(dcm_images, dcm_labels):
                self.train_images.append(imresize(dicom.read_file(img).pixel_array, [IMAGE_SIZE, IMAGE_SIZE]))
                self.train_labels.append(imresize(dicom.read_file(lbl).pixel_array, [IMAGE_SIZE, IMAGE_SIZE]))
        else:
            print "Train directory has not been set. Nothing to load."

    def load_val_data(self):
        if self.val_dir:
            dcm_images = [os.path.join(path, f)
                          for path, _, files in os.walk(self.val_dir)
                          for f in fnmatch.filter(files, 'image*.dcm')]

            dcm_labels = [os.path.join(path, f)
                          for path, _, files in os.walk(self.val_dir)
                          for f in fnmatch.filter(files, 'ribsMask*.dcm')]

            for img, lbl in zip(dcm_images, dcm_labels):
                self.val_images.append(imresize(dicom.read_file(img).pixel_array, [IMAGE_SIZE, IMAGE_SIZE]))
                self.val_labels.append(imresize(dicom.read_file(lbl).pixel_array, [IMAGE_SIZE, IMAGE_SIZE]))
        else:
            print "Val directory has not been set. Nothing to load."

    def train(self, iterations=10000):
        """
        Does training using train data within the object. Changes state of object and also saves progress.
        :param iterations: number of steps of gradient descent
        :return: nothing to return.
        """
        if not self.session_started:
            self.start_session()

        if len(self.train_images) * len(self.train_labels) == 0:
            print "There is no data within the object."
            return

        assert len(self.train_images) == len(self.train_labels)

        for i in range(iterations):
            indices = np.random.randint(0, len(self.train_images), BATCH_SIZE)
            batch_images = np.reshape(self.train_images[indices], [-1, np.square(IMAGE_SIZE)])
            batch_labels = np.reshape(self.train_labels[indices], [-1, np.square(IMAGE_SIZE)])
            if i % 10 == 0:
                train_accuracy = self.session.run(self.error,
                                                  feed_dict={self.x: batch_images, self.y_: batch_labels})
                print "step %d, training accuracy %g" % (i, train_accuracy)

            self.session.run(self.train_step, feed_dict={self.x: batch_images, self.y_: batch_labels})

        final_accuracy = self.session.run(self.error, feed_dict={self.x: self.val_images, self.y_: self.val_labels})
        print "Final error %g" % final_accuracy

        self.save_path = self.saver.save(self.session, "./fcnn_model.ckpt")

    def predict(self, image):
        """
        Does prediction according to the current state of the object.
        :return: A mask for an image given
        """
        if not self.session_started:
            self.start_session()

        return self.session.run(self.y, feed_dict={self.x: [image]})

    def load_model(self, model_path):
        if not self.session_started:
            self.start_session()

        self.saver.restore(self.session, model_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()


if __name__ == "__main__":
    with FCNN() as network:
        network.set_train_dir("./data/train")
        network.set_val_dir("./data_validate")
        network.train()
