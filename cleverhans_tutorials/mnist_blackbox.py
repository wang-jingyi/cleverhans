"""
This tutorial shows how to generate adversarial examples
using FGSM in black-box setting.
The original paper can be found at:
https://arxiv.org/abs/1602.02697
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange

import logging
import tensorflow as tf
from tensorflow.python.platform import flags

import os
import sys
sys.path.append('/Users/pxzhang/Documents/SUTD/project/deepxplore')
import random
import math
from scipy.misc import imsave

from MNIST_mine import utils
from MNIST_mine.gen_mutation import MutationTest

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils import to_categorical
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train, model_eval, batch_eval, model_argmax
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation

from cleverhans_tutorials.tutorial_models import make_basic_cnn, MLP
from cleverhans_tutorials.tutorial_models import Flatten, Linear, ReLU, Softmax
from cleverhans.utils import TemporaryLogLevel

FLAGS = flags.FLAGS


def setup_tutorial():
    """
    Helper function to check correct configuration of tf for tutorial
    :return: True if setup checks completed
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    return True


def prep_bbox(trained, sess, x, y, X_train, Y_train, X_test, Y_test,
              nb_epochs, batch_size, learning_rate,
              rng):
    """
    Define and train a model that simulates the "remote"
    black-box oracle described in the original paper.
    :param sess: the TF session
    :param x: the input placeholder for MNIST
    :param y: the ouput placeholder for MNIST
    :param X_train: the training data for the oracle
    :param Y_train: the training labels for the oracle
    :param X_test: the testing data for the oracle
    :param Y_test: the testing labels for the oracle
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param rng: numpy.random.RandomState
    :return:
    """

    # Define TF model graph (for the black-box model)
    model = make_basic_cnn()
    predictions = model(x)
    print("Defined TensorFlow model graph.")

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': 'mnist_bb/model',
        'filename': 'blackbox.model'
    }

    if trained:
        saver = tf.train.Saver()
        saver.restore(
            sess, os.path.join(
                train_params['train_dir'], train_params['filename']))
    else:
        model_train(sess, x, y, predictions, X_train, Y_train,
                    args=train_params, rng=rng, save=True)

    # Print out the accuracy on legitimate data
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params)
    print('Test accuracy of black-box on legitimate test '
          'examples: ' + str(accuracy))

    return model, predictions, accuracy


def substitute_model(img_rows=28, img_cols=28, nb_classes=10):
    """
    Defines the model architecture to be used by the substitute. Use
    the example model interface.
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: tensorflow model
    """
    input_shape = (None, img_rows, img_cols, 1)

    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(),
              Linear(200),
              ReLU(),
              Linear(200),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    return MLP(layers, input_shape)


def train_sub(trained, sess, x, y, bbox_preds, X_sub, Y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              rng):
    """
    This function creates the substitute by alternatively
    augmenting the training data and training the substitute.
    :param sess: TF session
    :param x: input TF placeholder
    :param y: output TF placeholder
    :param bbox_preds: output of black-box model predictions
    :param X_sub: initial substitute training data
    :param Y_sub: initial substitute training labels
    :param nb_classes: number of output classes
    :param nb_epochs_s: number of epochs to train substitute model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param data_aug: number of times substitute training data is augmented
    :param lmbda: lambda from arxiv.org/abs/1602.02697
    :param rng: numpy.random.RandomState instance
    :return:
    """
    # Define TF model graph (for the black-box model)
    model_sub = substitute_model()
    preds_sub = model_sub(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)

    # Train the substitute and augment dataset alternatively
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs_s,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'train_dir': 'mnist_bb/model',
            'filename': 'substitute.model'
        }
        with TemporaryLogLevel(logging.WARNING, "cleverhans.utils.tf"):
            if trained:
                saver = tf.train.Saver()
                saver.restore(
                    sess, os.path.join(
                        train_params['train_dir'], train_params['filename']))
            else:
                model_train(sess, x, y, preds_sub, X_sub,
                            to_categorical(Y_sub, nb_classes),
                            init_all=False, args=train_params, rng=rng, save=True)

        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads,
                                          lmbda_coef * lmbda)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub)/2):]
            eval_params = {'batch_size': batch_size}
            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model
            Y_sub[int(len(X_sub)/2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def mnist_blackbox(trained = True, mutated = True, train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_classes=10, batch_size=128,
                   learning_rate=0.001, nb_epochs=10, holdout=150, data_aug=6,
                   nb_epochs_s=10, lmbda=0.1,
                   attack_num=100):
    """
    MNIST tutorial for the black-box attack from arxiv.org/abs/1602.02697
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: a dictionary with:
             * black-box model accuracy on test set
             * substitute model accuracy on test set
             * black-box model accuracy on adversarial examples transferred
               from the substitute model
    """

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Dictionary used to keep track and return key accuracies
    accuracies = {}

    # Perform tutorial setup
    assert setup_tutorial()

    # Create TF session
    sess = tf.Session()

    # Get MNIST data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Initialize substitute training set reserved for adversary
    X_sub = X_test[:holdout]
    Y_sub = np.argmax(Y_test[:holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    X_test = X_test[holdout:]
    Y_test = Y_test[holdout:]

    # Define input and output TF placeholders
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Seed random number generator so tutorial is reproducible
    rng = np.random.RandomState([2017, 8, 30])

    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    print("Preparing the black-box model.")
    prep_bbox_out = prep_bbox(trained, sess, x, y, X_train, Y_train, X_test, Y_test,
                              nb_epochs, batch_size, learning_rate,
                              rng=rng)
    model, bbox_preds, accuracies['bbox'] = prep_bbox_out

    # Train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    train_sub_out = train_sub(trained, sess, x, y, bbox_preds, X_sub, Y_sub,
                              nb_classes, nb_epochs_s, batch_size,
                              learning_rate, data_aug, lmbda, rng=rng)
    model_sub, preds_sub = train_sub_out

    # Evaluate the substitute model on clean test examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_sub, X_test, Y_test, args=eval_params)
    accuracies['sub'] = acc

    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    fgsm_par = {'eps': 0.3, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    fgsm = FastGradientMethod(model_sub, sess=sess)

    # Craft adversarial examples using the substitute
    eval_params = {'batch_size': batch_size}
    x_adv_sub = fgsm.generate(x, **fgsm_par)

    # index = random.randint(0, len(Y_test) - attack_num)
    index = 0
    adv = sess.run(x_adv_sub, feed_dict={x: X_test[index: index + attack_num], y: Y_test[index: index + attack_num]})

    store_path = './mnist_bb/adv_blackbox'
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    labels = np.argmax(Y_test[index: index + attack_num], axis=1)
    new_class_labels = model_argmax(sess, x, model(x), adv)
    for i in range(len(adv)):
        # if labels[i] != new_class_labels[i]:
        adv_img_deprocessed = utils.deprocess_image(np.asarray([adv[i]]))
        imsave(store_path + '/adv_' + str(i) + '_' + str(labels[i]) + '_' + str(
            new_class_labels[i]) + '_.png', adv_img_deprocessed)

    path = './mnist_bb'
    result = ''

    [image_list, image_files, real_labels, predicted_labels] = utils.get_data_mutation_test(store_path)
    img_rows = 28
    img_cols = 28
    seed_number = len(image_list)
    mutation_number = 1000

    mutation_test = MutationTest(img_rows, img_cols, seed_number, mutation_number)
    mutations = []
    if mutated:
        mutations = np.load(path + "/mutation_list.npy")
    else:
        for i in range(mutation_number):
            mutation = mutation_test.mutation_matrix()
            mutations.append(mutation)
        np.save(path + "/mutation_list.npy", mutations)

    store_string = ''
    for step_size in [1, 5, 10]:

        label_change_numbers = []
        # Iterate over all the test data
        for i in range(len(image_list)):
            ori_img = np.expand_dims(image_list[i], axis=2)
            ori_img = ori_img.astype('float32')
            ori_img /= 255
            orig_label = predicted_labels[i]

            label_changes = 0
            for j in range(mutation_number):
                img = ori_img.copy()
                add_mutation = mutations[j][0]
                mu_img = img + add_mutation * step_size

                # Predict the label for the mutation
                mu_img = np.expand_dims(mu_img, 0)
                # print(mu_img.shape)
                # Define input placeholder
                # input_x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))

                mu_label = model_argmax(sess, x, model(x), mu_img)
                # print('Predicted label: ', mu_label)

                if mu_label != int(orig_label):
                    label_changes += 1

            label_change_numbers.append(label_changes)
            store_string = store_string + image_files[i] + "," + str(step_size) + "," + str(label_changes) + "\n"

        label_change_numbers = np.asarray(label_change_numbers)
        adv_average = round(np.mean(label_change_numbers), 2)
        adv_std = np.std(label_change_numbers)
        adv_95ci = round(1.96 * adv_std / math.sqrt(len(label_change_numbers)), 2)
        result = result + 'adv,' + str(step_size) + ',' + str(adv_average) + ',' + str(
            round(adv_std, 2)) + ',' + str(adv_95ci) + '\n'

        # print('Number of label changes for step size: ' + str(step_size) + ', ' + str(label_change_numbers))
    with open(path + "/adv_result.csv", "w") as f:
        f.write(store_string)

    store_path = './mnist_bb/ori_blackbox'
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    choice = []
    image_list = []
    predicted_labels = []
    real_labels = []
    while len(choice) != 500:
        index = random.randint(0, len(X_test) - 1)
        if index not in choice:
            choice.append(index)
            image_list.append(X_test[index])
            real_labels.append(Y_test[index])

    np.save(store_path + '/ori_x.npy', np.asarray(image_list))
    np.save(store_path + '/ori_y.npy', np.asarray(real_labels))

    image_list = np.load(store_path + '/ori_x.npy')
    real_labels = np.load(store_path + '/ori_y.npy')
    seed_number = len(image_list)
    mutation_number = 1000

    store_string = ''
    for step_size in [1, 5, 10]:

        label_change_numbers = []
        # Iterate over all the test data
        for i in range(len(image_list)):
            ori_img = image_list[i]
            orig_label = model_argmax(sess, x, model(x), np.asarray([image_list[i]]))

            label_changes = 0
            for j in range(mutation_number):
                img = ori_img.copy()
                add_mutation = mutations[j][0]
                mu_img = img + add_mutation * step_size

                # Predict the label for the mutation
                mu_img = np.expand_dims(mu_img, 0)

                mu_label = model_argmax(sess, x, model(x), mu_img)
                # print('Predicted label: ', mu_label)

                if mu_label != int(orig_label):
                    label_changes += 1

            label_change_numbers.append(label_changes)
            store_string = store_string + str(i) + "," + str(step_size) + "," + str(label_changes) + "\n"

        label_change_numbers = np.asarray(label_change_numbers)
        adv_average = round(np.mean(label_change_numbers), 2)
        adv_std = np.std(label_change_numbers)
        adv_95ci = round(1.96 * adv_std / math.sqrt(len(label_change_numbers)), 2)
        result = result + 'ori,' + str(step_size) + ',' + str(adv_average) + ',' + str(
            round(adv_std, 2)) + ',' + str(adv_95ci) + '\n'
        # print('Number of label changes for step size: ' + str(step_size)+ ', '+ str(label_change_numbers))

    with open(path + "/ori_result.csv", "w") as f:
        f.write(store_string)

    with open(path + "/result.csv", "w") as f:
        f.write(result)

    # Evaluate the accuracy of the "black-box" model on adversarial examples
    accuracy = model_eval(sess, x, y, model(x_adv_sub), X_test, Y_test,
                          args=eval_params)
    print('Test accuracy of oracle on adversarial examples generated '
          'using the substitute: ' + str(accuracy))
    accuracies['bbox_on_sub_adv_ex'] = accuracy

    return accuracies


def main(argv=None):
    mnist_blackbox(trained = FLAGS.trained,
                   mutated=FLAGS.mutated,
                   attack_num=FLAGS.attack_num,
                   nb_classes=FLAGS.nb_classes, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                   data_aug=FLAGS.data_aug, nb_epochs_s=FLAGS.nb_epochs_s,
                   lmbda=FLAGS.lmbda)


if __name__ == '__main__':
    # General flags
    flags.DEFINE_boolean('trained', False, 'The model is already trained.')  # default:False
    flags.DEFINE_boolean('mutated', False, 'The mutation list is already generate.')  # default:False
    flags.DEFINE_integer('attack_num', 500, 'The number of original data to attack.')
    flags.DEFINE_integer('nb_classes', 10, 'Number of classes in problem')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    # Flags related to oracle
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model')

    # Flags related to substitute
    flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary')
    flags.DEFINE_integer('data_aug', 6, 'Nb of substitute data augmentations')
    flags.DEFINE_integer('nb_epochs_s', 10, 'Training epochs for substitute')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')

    tf.app.run()
