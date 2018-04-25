"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging
from scipy.misc import imsave

import os
import random
import sys
sys.path.append('/Users/pxzhang/Documents/SUTD/project/deepxplore')

from MNIST_mine import utils
from MNIST_mine.gen_mutation import MutationTest

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, model_argmax
from cleverhans.attacks import FastGradientMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level

import os

FLAGS = flags.FLAGS


def mnist_tutorial(trained = True, train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64, num_threads=None,
                   attack_num=100):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': 'model',
        'filename': 'fgsm_1.model'
    }
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    rng = np.random.RandomState([2017, 8, 30])

    if clean_train:
        model = make_basic_cnn(nb_filters=nb_filters)
        preds = model.get_probs(x)
        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)

        if trained:
            saver = tf.train.Saver()
            saver.restore(
                sess, os.path.join(
                    train_params['train_dir'], train_params['filename']))
        else:
            model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                        args=train_params, rng=rng,save=True)

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        fgsm = FastGradientMethod(model, sess=sess)
        adv_x = fgsm.generate(x, **fgsm_params)
        preds_adv = model.get_probs(adv_x)

        index = random.randint(test_start, test_end - attack_num)
        adv= sess.run(adv_x,feed_dict={x: X_test[index: index + 100], y: Y_test[index: index + 100]})

        store_path = './adv_fgsm1'
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        labels = np.argmax(Y_test[index: index + 100], axis=1)
        new_class_labels = model_argmax(sess, x, preds, adv)
        for i in range(len(adv)):
            if labels[i] != new_class_labels[i]:
                adv_img_deprocessed = utils.deprocess_image(np.asarray([adv[i]]))
                imsave(store_path + '/' + str(i) +'adv_' + str(labels[i]) + '_' + str(
                    new_class_labels[i]) + '_.png', adv_img_deprocessed)

        [image_list, real_labels, predicted_labels] = utils.get_data_mutation_test(
            '/Users/pxzhang/Documents/SUTD/project/cleverhans/cleverhans_tutorials/adv_fgsm1')
        img_rows = 28
        img_cols = 28
        seed_number = len(predicted_labels)
        mutation_number = 1000

        mutation_test = MutationTest(img_rows, img_cols, seed_number, mutation_number)
        mutations = []
        for i in range(mutation_number):
            mutation = mutation_test.mutation_matrix()
            mutations.append(mutation)

        for step_size in [1, 5, 10]:

            label_change_numbers = []
            # Iterate over all the test data
            for i in range(len(predicted_labels)):
                ori_img = np.expand_dims(image_list[i], axis=2)
                ori_img = ori_img.astype('float32')
                ori_img /= 255
                orig_label = predicted_labels[i]

                label_changes = 0
                for j in range(mutation_test.mutation_number):
                    img = ori_img.copy()
                    add_mutation = mutations[j][0]
                    mu_img = img + add_mutation * step_size

                    # Predict the label for the mutation
                    mu_img = np.expand_dims(mu_img, 0)
                    # print(mu_img.shape)
                    # Define input placeholder
                    # input_x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))

                    mu_label = model_argmax(sess, x, preds, mu_img)
                    # print('Predicted label: ', mu_label)

                    if mu_label != orig_label:
                        label_changes += 1

                label_change_numbers.append(label_changes)

            print('Number of label changes for step size: ' + str(step_size) + ', ' + str(label_change_numbers))


        # # Evaluate the accuracy of the MNIST model on adversarial examples
        # eval_par = {'batch_size': batch_size}
        # acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        # print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        # report.clean_train_adv_eval = acc

        # # Calculate training error
        # if testing:
        #     eval_par = {'batch_size': batch_size}
        #     acc = model_eval(sess, x, y, preds_adv, X_train,
        #                      Y_train, args=eval_par)
        #     report.train_clean_train_adv_eval = acc

        print("Repeating the process, using adversarial training")
    # Redefine TF model graph
    model_2 = make_basic_cnn(nb_filters=nb_filters)
    preds_2 = model_2(x)
    fgsm2 = FastGradientMethod(model_2, sess=sess)
    adv_x_2 = fgsm2.generate(x, **fgsm_params)
    if not backprop_through_attack:
        # For the fgsm attack used in this tutorial, the attack has zero
        # gradient so enabling this flag does not change the gradient.
        # For some other attacks, enabling this flag increases the cost of
        # training, but gives the defender the ability to anticipate how
        # the atacker will change their strategy in response to updates to
        # the defender's parameters.
        adv_x_2 = tf.stop_gradient(adv_x_2)
    preds_2_adv = model_2(adv_x_2)

    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_test, Y_test,
                              args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.adv_train_clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, X_test,
                              Y_test, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        report.adv_train_adv_eval = accuracy

    # Perform and evaluate adversarial training
    train_params['filename'] = 'fgsm_2.model'
    if trained:
        saver = tf.train.Saver()
        saver.restore(
            sess, os.path.join(
                train_params['train_dir'], train_params['filename']))
    else:
        model_train(sess, x, y, preds_2, X_train, Y_train,
                   predictions_adv=preds_2_adv, evaluate=evaluate_2,
                    args=train_params, rng=rng,save=True)

    adv2 = sess.run(adv_x_2, feed_dict={x: X_test[index: index + 100], y: Y_test[index: index + 100]})

    store_path = './adv_fgsm2'
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    new_class_labels = model_argmax(sess, x, preds_2, adv2)
    for i in range(len(adv2)):
        if labels[i] != new_class_labels[i]:
            adv_img_deprocessed = utils.deprocess_image(np.asarray([adv2[i]]))
            imsave(store_path + '/' + str(i) + 'adv_' + str(labels[i]) + '_' + str(
                new_class_labels[i]) + '_.png', adv_img_deprocessed)

    [image_list, real_labels, predicted_labels] = utils.get_data_mutation_test(
        '/Users/pxzhang/Documents/SUTD/project/cleverhans/cleverhans_tutorials/adv_fgsm2')
    # img_rows = 28
    # img_cols = 28
    # seed_number = len(predicted_labels)
    # mutation_number = 1000
    #
    # mutation_test = MutationTest(img_rows, img_cols, seed_number, mutation_number)
    # mutations = []
    # for i in range(mutation_number):
    #     mutation = mutation_test.mutation_matrix()
    #     mutations.append(mutation)

    for step_size in [1, 5, 10]:

        label_change_numbers = []
        # Iterate over all the test data
        for i in range(len(predicted_labels)):
            ori_img = np.expand_dims(image_list[i], axis=2)
            ori_img = ori_img.astype('float32')
            ori_img /= 255
            orig_label = predicted_labels[i]

            label_changes = 0
            for j in range(mutation_test.mutation_number):
                img = ori_img.copy()
                add_mutation = mutations[j][0]
                mu_img = img + add_mutation * step_size

                # Predict the label for the mutation
                mu_img = np.expand_dims(mu_img, 0)
                # print(mu_img.shape)
                # Define input placeholder
                # input_x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))

                mu_label = model_argmax(sess, x, preds, mu_img)
                # print('Predicted label: ', mu_label)

                if mu_label != orig_label:
                    label_changes += 1

            label_change_numbers.append(label_changes)

        print('Number of label changes for step size: ' + str(step_size) + ', ' + str(label_change_numbers))

    # # Calculate training errors
    # if testing:
    #     eval_params = {'batch_size': batch_size}
    #     accuracy = model_eval(sess, x, y, preds_2, X_train, Y_train,
    #                           args=eval_params)
    #     report.train_adv_train_clean_eval = accuracy
    #     accuracy = model_eval(sess, x, y, preds_2_adv, X_train,
    #                           Y_train, args=eval_params)
    #     report.train_adv_train_adv_eval = accuracy

    return report


def main(argv=None):
    mnist_tutorial(trained = FLAGS.trained,
                   attack_num = FLAGS.attack_num,
                   nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_boolean('trained', False, 'The model is already trained.')  # default:False
    flags.DEFINE_integer('attack_num', 100, 'The number of original data to attack.')
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    tf.app.run()
