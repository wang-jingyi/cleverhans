"""
This tutorial shows how to generate adversarial examples
using JSMA in white-box setting.
The original paper can be found at:
https://arxiv.org/abs/1511.07528
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import logging
from scipy.misc import imsave

import os
import sys

from MNIST_mine import utils
from MNIST_mine.gen_mutation import MutationTest
# from utils import deprocess_mnist_image

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_cifar10 import data_cifar10, deprocess_image, preprocess_image
from cleverhans.utils_tf import model_train, model_eval, model_argmax
from cleverhans.utils_keras import KerasModelWrapper, cnn_model
from cleverhans_tutorials.tutorial_models import make_basic_cnn_cifar10
from os.path import expanduser
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import FastGradientMethod
FLAGS = flags.FLAGS

def cifar10_tutorial_tf(trained = True, mutated = True, train_start=0, train_end=50000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64, num_threads=None,
                   attack_num=100):
    """
    CIFAR10 tutorial for Carlini and Wagner's attack (CW)
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # CIFAR10-specific dimensions
    img_rows = 32
    img_cols = 32
    channels = 3

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get CIFAR10 test data
    X_train, Y_train, fn_train, X_test, Y_test, fn_test = data_cifar10(train_start=train_start,
                                                                       train_end=train_end,
                                                                       test_start=test_start,
                                                                       test_end=test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = make_basic_cnn_cifar10()
    preds = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################
    # save_file = './MNIST_trained_model_jsma'
    # saver = tf.train.Saver()

    # if os.path.exists('./MNIST_trained_model_jsma.meta'):
    #     # Restore the trained model
    #     print('Restore trained model ...')
    #     new_saver = tf.train.import_meta_graph('./MNIST_trained_model_jsma.meta')
    #     new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    #
    # else:
        # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': 'cifar10/model',
        'filename': 'jsma.model'
    }

    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}

    if trained:
        saver = tf.train.Saver()
        saver.restore(
            sess, os.path.join(
                train_params['train_dir'], train_params['filename']))
    else:
        sess.run(tf.global_variables_initializer())
        rng = np.random.RandomState([2017, 8, 30])
        model_train(sess, x, y, preds, X_train, Y_train, args=train_params,
                    rng=rng,save=True)

    # Save the trained model to avoid retrain
    # saver.save(sess, save_file)
    # print('Trained model saved to MNIST_trained_model_jsma.ckpt ...')

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using the FGSM approach
    ###########################################################################
    # Initialize the Fast Gradient Sign Method (FGSM) attack object and
    # graph
    fgsm = FastGradientMethod(model, sess=sess)
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model.get_probs(adv_x)

    # index = random.randint(test_start, test_end - attack_num)
    index = 0
    adv = sess.run(adv_x, feed_dict={x: X_test[index: index + attack_num], y: Y_test[index: index + attack_num]})

    store_path = './cifar10_tf/adv_fgsm'
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    labels = np.argmax(Y_test[index: index + attack_num], axis=1)
    new_class_labels = model_argmax(sess, x, preds, adv)
    for i in range(len(adv)):
        # if labels[i] != new_class_labels[i]:
        adv_img_deprocessed = deprocess_image(np.asarray([adv[i]]))
        imsave(store_path + '/adv_' + str(i) + '_' + str(labels[i]) + '_' + str(
            new_class_labels[i]) + '_.png', adv_img_deprocessed)

    print('--------------------------------------')

    # Generate random matution matrix for mutations
    home = expanduser("~")
    [image_list, image_files, real_labels, predicted_labels] = utils.get_data_mutation_test(home + '/cleverhans/cleverhans_tutorials/cifar10_tf/adv_fgsm')
    img_rows = 32
    img_cols = 32
    seed_number = len(predicted_labels)
    mutation_number = 1000

    mutation_test = MutationTest(img_rows, img_cols, seed_number, mutation_number)
    mutations = []
    for i in range(mutation_number):
        mutation = mutation_test.mutation_matrix()
        mutations.append(mutation)

    for step_size in [1,5,10]:

        label_change_numbers = []
        # Iterate over all the test data
        for i in range(len(predicted_labels)):
            ori_img = preprocess_image(image_list[i].astype('float64'))
            orig_label = predicted_labels[i]

            label_changes = 0
            for j in range(mutation_test.mutation_number):
                img = ori_img.copy()
                add_mutation = mutations[j][0]
                mu_img = img + add_mutation * step_size
                # print("----------------------------------------")
                # print(add_mutation)

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

        print('Number of label changes for step size: ' + str(step_size)+ ', '+ str(label_change_numbers))

    # # Compute the number of adversarial examples that were successfully found
    # nb_targets_tried = ((nb_classes - 1) * source_samples)
    # succ_rate = float(np.sum(results)) / nb_targets_tried
    # print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))
    # report.clean_train_adv_eval = 1. - succ_rate
    #
    # # Compute the average distortion introduced by the algorithm
    # percent_perturbed = np.mean(perturbations)
    # print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))
    #
    # # Compute the average distortion introduced for successful samples only
    # percent_perturb_succ = np.mean(perturbations * (results == 1))
    # print('Avg. rate of perturbed features for successful '
    #       'adversarial examples {0:.4f}'.format(percent_perturb_succ))

    # Close TF session
    sess.close()
    print('Finish generating adversaries.')

    # Finally, block & display a grid of all the adversarial examples
    # if viz_enabled:
    #     import matplotlib.pyplot as plt
    #     plt.close(figure)
    #     _ = grid_visual(grid_viz_data)

    return report




def main(argv=None):
    cifar10_tutorial_tf(trained = FLAGS.trained,
                   mutated=FLAGS.mutated,
                   attack_num = FLAGS.attack_num,
                   nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_boolean('trained', True, 'The model is already trained.')  # default:False
    flags.DEFINE_boolean('mutated', False, 'The mutation list is already generate.')  # default:False
    flags.DEFINE_integer('attack_num', 5, 'The number of original data to attack.')
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    tf.app.run()
