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
from cleverhans.utils_cifar10 import data_cifar10, deprocess_image, preprocess_image
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

FLAGS = flags.FLAGS

def cifar10_tutorial_cw(trained = True, mutated = True, train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=True, nb_epochs=300,
                      batch_size=128, nb_classes=10, source_samples=10,
                      learning_rate=0.001, attack_iterations=100,
                      model_path=os.path.join("models", "cifar10"),
                      targeted=True):
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
    # Craft adversarial examples using the CW approach
    ###########################################################################
    print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes-1) +
          ' adversarial examples')

    nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
    print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
          ' adversarial examples')
    print("This could take some time ...")

    # Instantiate a CW attack object
    cw = CarliniWagnerL2(model, back='tf', sess=sess)

    if viz_enabled:
        # assert source_samples == nb_classes
        # idxs = [np.where(np.argmax(Y_test, axis=1) == i)[0][0]
        #         for i in range(nb_classes)]
        temp = np.argmax(Y_test, axis=1)
        idxs = []
        for i in range(source_samples):
            idx = np.where(temp == i % 10)[0][0]
            idxs.append(idx)
            temp[idx] = 10

    if targeted:
        if viz_enabled:
            # Initialize our array for grid visualization
            grid_shape = (nb_classes, nb_classes, img_rows, img_cols, channels)
            grid_viz_data = np.zeros(grid_shape, dtype='f')

            adv_inputs = np.array(
                [[instance] * nb_classes for instance in X_test[idxs]],
                dtype=np.float32)
            label = np.array([[labels] * nb_classes for labels in Y_test[idxs]], dtype=np.int)
        else:
            adv_inputs = np.array(
                [[instance] * nb_classes for
                 instance in X_test[:source_samples]], dtype=np.float32)

        one_hot = np.zeros((nb_classes, nb_classes))
        one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1

        adv_inputs = adv_inputs.reshape(
            (source_samples * nb_classes, img_rows, img_cols, 3))
        adv_ys = np.array([one_hot] * source_samples,
                          dtype=np.float32).reshape((source_samples *
                                                     nb_classes, nb_classes))
        yname = "y_target"
    else:
        if viz_enabled:
            # Initialize our array for grid visualization
            grid_shape = (nb_classes, 2, img_rows, img_cols, channels)
            grid_viz_data = np.zeros(grid_shape, dtype='f')

            adv_inputs = X_test[idxs]
        else:
            adv_inputs = X_test[:source_samples]

        adv_ys = None
        yname = "y"

    cw_params = {'binary_search_steps': 1,
                 yname: adv_ys,
                 'max_iterations': attack_iterations,
                 'learning_rate': 0.1,
                 'batch_size': source_samples * nb_classes if
                 targeted else source_samples,
                 'initial_const': 10}

    adv = cw.generate_np(adv_inputs, **cw_params)

    targets = np.argmax(adv_ys, axis=1)

    store_path = './cifar10_cw/adv_cw'
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    new_class_labels = model_argmax(sess, x, preds, adv)

    for i in range(len(adv)):
        #      Predicted class of the generated adversary
        res = int(new_class_labels[i] == targets[i])
        if res == 1:
            adv_img_deprocessed = deprocess_image(np.asarray([adv[i]]))
            imsave(
                store_path + '/adv_' + str(i) + '_' + str(np.argmax(label[int(i / 10)][int(i % 10)], axis=0)) + '_' + str(
                    new_class_labels[i]) + '_.png',
                adv_img_deprocessed)

    print('--------------------------------------')

    # Generate random matution matrix for mutations
    home = expanduser("~")
    [image_list, real_labels, predicted_labels] = utils.get_data_mutation_test(home + '/cleverhans/cleverhans_tutorials/cifar10_adv_cw')
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
    cifar10_tutorial_cw(trained = FLAGS.trained,
                      mutated=FLAGS.mutated,
                      viz_enabled=FLAGS.viz_enabled,
                      nb_epochs=FLAGS.nb_epochs,
                      batch_size=FLAGS.batch_size,
                      nb_classes=FLAGS.nb_classes,
                      source_samples=FLAGS.source_samples,
                      learning_rate=FLAGS.learning_rate,
                      attack_iterations=FLAGS.attack_iterations,
                      model_path=FLAGS.model_path,
                      targeted=FLAGS.targeted)


if __name__ == '__main__':
    flags.DEFINE_boolean('trained', True, 'The model is already trained.')  # default:False
    flags.DEFINE_boolean('mutated', False, 'The mutation list is already generate.')  # default:False
    flags.DEFINE_boolean('viz_enabled', False, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 50, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('model_path', os.path.join("cifar10/model", "cifar10"),
                        'Path to save or load the model file')
    flags.DEFINE_integer('attack_iterations', 100,
                         'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', True,
                         'Run the tutorial in targeted mode?')

    tf.app.run()
