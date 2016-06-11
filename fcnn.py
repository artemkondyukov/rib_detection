from matplotlib import pyplot as plt
from scipy.misc import imresize, toimage

import dicom
import fnmatch
import numpy as np
import os
import sys

from initializer import *

# train and validate images will be resized to square with side
IMAGE_SIZE = 256
BATCH_SIZE = 5


def convolution2d(x_img, W_conv):
    return tf.nn.conv2d(x_img, W_conv, strides=[1, 1, 1, 1], padding='SAME')


def deconvolution2d(x_img, W_conv, output_shape, strides=None):
    if strides is None:
        strides = [1, 1, 1, 1]
    return tf.nn.conv2d_transpose(x_img, W_conv, output_shape=output_shape, strides=strides, padding='SAME')


def max_pool_2x2(x_img):
    return tf.nn.max_pool(x_img, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class FCNN:
    def __init__(self, train_dir=None, val_dir=None, _lambda=None):
        self.train_dir = train_dir
        self.val_dir = val_dir

        if _lambda is None:
            self._lambda = 3e-3
        else:
            self._lambda = _lambda

        self.train_images = np.empty(0)
        self.train_labels = np.empty(0)

        self.val_images = np.empty(0)
        self.val_labels = np.empty(0)

        # Convolution
        self.W_conv1 = weight_variable([5, 5, 1, 8], "W_conv1")
        self.b_conv1 = bias_variable([8], "b_conv1")

        self.W_conv2 = weight_variable([5, 5, 8, 16], "W_conv2")
        self.b_conv2 = bias_variable([16], "b_conv2")

        self.W_conv3 = weight_variable([3, 3, 16, 5], "W_conv3")
        self.b_conv3 = bias_variable([5], "b_conv3")

        self.W_conv4 = weight_variable([3, 3, 16, 5], "W_conv4")
        self.b_conv4 = bias_variable([5], "b_conv4")

        self.x = tf.placeholder(tf.float32, shape=[None, np.square(IMAGE_SIZE)])
        self.y_ = tf.placeholder(tf.float32, shape=[None, np.square(IMAGE_SIZE)])

        self.x_img = tf.reshape(self.x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

        self.h_conv1 = tf.nn.relu(convolution2d(self.x_img, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)
        self.h_conv2 = tf.nn.relu(convolution2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)
        self.h_conv3 = tf.nn.relu(convolution2d(self.h_pool2, self.W_conv3) + self.b_conv3)

        # Deconvolution
        current_batch_size = tf.shape(self.x)[0]
        self.W_deconv4 = weight_variable([3, 3, 16, 5], "W_deconv4")
        self.b_deconv4 = bias_variable([16], "b_deconv4")
        self.deconv4_shape = tf.pack([current_batch_size, 64, 64, 16])

        self.W_pool3 = weight_variable([2, 2, 16, 16], "W_pool3")
        self.b_pool3 = bias_variable([16], "b_pool3")
        self.pool3_shape = tf.pack([current_batch_size, 64, 64, 16])

        self.W_deconv3 = weight_variable([3, 3, 16, 16], "W_deconv3")
        self.b_deconv3 = bias_variable([16], "b_deconv3")
        self.deconv3_shape = tf.pack([current_batch_size, 128, 128, 16])

        self.W_pool2 = weight_variable([2, 2, 16, 16], "W_pool2")
        self.b_pool2 = bias_variable([16], "b_pool2")
        self.pool2_shape = tf.pack([current_batch_size, 128, 128, 16])

        self.W_deconv2 = weight_variable([5, 5, 8, 16], "W_deconv2")
        self.b_deconv2 = bias_variable([8], "b_deconv2")
        self.deconv2_shape = tf.pack([current_batch_size, 256, 256, 8])

        self.W_pool1 = weight_variable([2, 2, 8, 8], "W_pool1")
        self.b_pool1 = bias_variable([8], "b_pool1")
        self.pool1_shape = tf.pack([current_batch_size, 256, 256, 8])

        self.W_deconv1 = weight_variable([5, 5, 1, 8], "W_deconv1")
        self.b_deconv1 = bias_variable([1], "b_deconv1")
        self.deconv1_shape = tf.pack([current_batch_size, 256, 256, 1])

        self.h_deconv4 = tf.nn.relu(
            deconvolution2d(
                tf.image.resize_images(self.h_conv3, IMAGE_SIZE / 4, IMAGE_SIZE / 4),
                self.W_deconv4, self.deconv4_shape) + self.b_deconv4)
        self.h_deconv3 = tf.nn.relu(
            deconvolution2d(
                tf.image.resize_images(self.h_deconv4, IMAGE_SIZE / 2, IMAGE_SIZE / 2),
                self.W_deconv3, self.deconv3_shape) + self.b_deconv3)
        self.h_deconv2 = tf.nn.relu(
            deconvolution2d(
                tf.image.resize_images(self.h_deconv3, IMAGE_SIZE, IMAGE_SIZE),
                self.W_deconv2, self.deconv2_shape) + self.b_deconv2)
        self.h_deconv1 = tf.nn.sigmoid(
            deconvolution2d(self.h_deconv2, self.W_deconv1, self.deconv1_shape) + self.b_deconv1)

        self.y = tf.reshape(self.h_deconv1, [-1, np.square(IMAGE_SIZE)])
        self.error = tf.reduce_mean(tf.square(self.y - self.y_)) + \
                     self._lambda / 2 * \
                     (
                         tf.reduce_mean(tf.square(self.W_conv1)) +
                         tf.reduce_mean(tf.square(self.W_conv2)) +
                         tf.reduce_mean(tf.square(self.W_conv3)) +
                         tf.reduce_mean(tf.square(self.W_conv4)) +
                         tf.reduce_mean(tf.square(self.W_deconv4)) +
                         tf.reduce_mean(tf.square(self.W_deconv3)) +
                         tf.reduce_mean(tf.square(self.W_deconv2)) +
                         tf.reduce_mean(tf.square(self.W_deconv1))
                     )
        self.train_step = tf.train.AdamOptimizer(1e-5).minimize(self.error)

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

            tmp_images = []
            tmp_labels = []
            for img, lbl in zip(dcm_images, dcm_labels):
                tmp_images.append(imresize(dicom.read_file(img).pixel_array, [IMAGE_SIZE, IMAGE_SIZE]))
                tmp_labels.append(imresize(dicom.read_file(lbl).pixel_array, [IMAGE_SIZE, IMAGE_SIZE]))
            self.train_images = np.array(tmp_images, dtype=np.float32) / np.max(tmp_images)
            self.train_labels = np.array(tmp_labels, dtype=np.float32) / np.max(tmp_labels)
        else:
            print "Train directory has not been set. Nothing to load."

    def load_val_data(self):
        if self.val_dir:
            dcm_images = sorted([os.path.join(path, f)
                                 for path, _, files in os.walk(self.val_dir)
                                 for f in fnmatch.filter(files, 'image*.dcm')])

            dcm_labels = sorted([os.path.join(path, f)
                                 for path, _, files in os.walk(self.val_dir)
                                 for f in fnmatch.filter(files, 'ribsMask*.dcm')])

            tmp_images = []
            tmp_labels = []
            for img, lbl in zip(dcm_images, dcm_labels):
                tmp_images.append(imresize(dicom.read_file(img).pixel_array, [IMAGE_SIZE, IMAGE_SIZE]))
                tmp_labels.append(imresize(dicom.read_file(lbl).pixel_array, [IMAGE_SIZE, IMAGE_SIZE]))
            self.val_images = np.array(tmp_images) / np.max(tmp_images)
            self.val_labels = np.array(tmp_labels) / np.max(tmp_labels)
        else:
            print "Val directory has not been set. Nothing to load."

    def train(self, iterations=10):
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
            if i % 100 == 0:
                train_accuracy = self.session.run(tf.reduce_mean(tf.abs(self.y - self.y_)),
                                                  feed_dict={self.x: batch_images, self.y_: batch_labels})
                print "step %d, training accuracy %g" % (i, train_accuracy)
                directory = './pics/' + str(self._lambda) + '_' + str(i) + "/"
                if not os.path.exists(directory):
                    os.makedirs(directory)

                prediction = self.session.run(self.y, feed_dict={self.x: [batch_images[0]]})
                toimage(np.reshape(batch_images[0], [IMAGE_SIZE, IMAGE_SIZE]), cmin=0., cmax=1.).\
                    save(directory + 'image.jpg')
                toimage(np.reshape(prediction, [IMAGE_SIZE, IMAGE_SIZE]), cmin=0., cmax=1.).\
                    save(directory + 'prediction.jpg')
                toimage(np.reshape(batch_labels[0], [IMAGE_SIZE, IMAGE_SIZE]), cmin=0., cmax=1.).\
                    save(directory + 'ground.jpg')
                # plt.set_cmap('gray')
                # plt.figure(0)
                # plt.imshow(np.reshape(batch_images[0], [IMAGE_SIZE, IMAGE_SIZE]))
                # plt.figure(1)
                # plt.imshow(np.reshape(self.session.run(self.y,
                #                                        feed_dict={self.x: [batch_images[0]]}),
                #                       [IMAGE_SIZE, IMAGE_SIZE]))
                # plt.figure(2)
                # plt.imshow(np.reshape(batch_labels[0], [IMAGE_SIZE, IMAGE_SIZE]))
                #
                # plt.show(block=True)

            if i % 1000 == 0:
                self.save_path = self.saver.save(self.session,
                                                 "./models/fcnn_model_" + str(self._lambda) + "_" + str(i) + ".ckpt")

            self.session.run(self.train_step, feed_dict={self.x: batch_images, self.y_: batch_labels})

        validate_images = np.reshape(self.val_images, [-1, np.square(IMAGE_SIZE)])
        validate_labels = np.reshape(self.val_labels, [-1, np.square(IMAGE_SIZE)])
        final_accuracy = self.session.run(tf.reduce_mean(tf.abs(self.y - self.y_)),
                                          feed_dict={self.x: validate_images, self.y_: validate_labels})
        print "Final error %g" % final_accuracy

        self.save_path = self.saver.save(self.session, "./models/fcnn_model_" + str(self._lambda) + ".ckpt")

    def predict(self, image):
        """
        Does prediction according to the current state of the object.
        :param image: an image to segment
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
    if len(sys.argv) == 1 or sys.argv[1] == 'train':
        for _lambda in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
            with FCNN(_lambda=_lambda) as network:

                network.set_train_dir("./data/train")
                network.set_val_dir("./data/validate")

                if len(sys.argv) < 3:
                    iterations = 10000
                else:
                    iterations = int(sys.argv[2])
                network.train(iterations=iterations)

    elif sys.argv[1] == 'predict':
        with FCNN() as network:
            if len(sys.argv) < 4:
                img_path = './data/validate/image1.dcm'
                lbl_path = './data/validate/ribsMask1.dcm'
            else:
                img_path = sys.argv[2]
                lbl_path = sys.argv[3]

            img = np.reshape(imresize(dicom.read_file(img_path).pixel_array,
                                      [IMAGE_SIZE, IMAGE_SIZE]), [np.square(IMAGE_SIZE)])
            img = np.array(img, dtype=np.float32) / img.max()
            lbl = np.reshape(imresize(dicom.read_file(lbl_path).pixel_array,
                                      [IMAGE_SIZE, IMAGE_SIZE]), [np.square(IMAGE_SIZE)])
            lbl = lbl / lbl.max()
            network.load_model('./fcnn_model_0.003_6000.ckpt')
            pred = network.predict(img)
            pred[pred < 0.5] = 0
            pred[pred != 0] = 1

            plt.set_cmap('gray')
            plt.figure(0)
            plt.imshow(np.reshape(img, [IMAGE_SIZE, IMAGE_SIZE]))
            plt.figure(1)
            plt.imshow(np.reshape(pred, [IMAGE_SIZE, IMAGE_SIZE]))
            plt.figure(2)
            plt.imshow(np.reshape(lbl, [IMAGE_SIZE, IMAGE_SIZE]))

            plt.show(block=True)
