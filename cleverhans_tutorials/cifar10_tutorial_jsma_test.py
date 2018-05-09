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

from MNIST_mine import utils
from MNIST_mine.gen_mutation import MutationTest
# from utils import deprocess_mnist_image

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_cifar10 import data_cifar10, deprocess_image, preprocess_image
from cleverhans.utils_tf import model_train, model_eval, model_argmax
from cleverhans.utils_keras import KerasModelWrapper, cnn_model
from cleverhans_tutorials.tutorial_models import make_basic_cnn_cifar10,make_better_cnn_cifar10
from os.path import expanduser

FLAGS = flags.FLAGS

def cifar10_tutorial_jsma(trained = True, train_start=0, train_end=50000, test_start=0,
                        test_end=10000, viz_enabled=True, nb_epochs=6,
                        batch_size=128, nb_classes=10, source_samples=10,
                        learning_rate=0.001):
    """
    CIFAR10 tutorial for the Jacobian-based saliency map approach (JSMA)
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
    model = make_better_cnn_cifar10()#make_basic_cnn_cifar10()
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
        'train_dir': 'cifar10_jsma/model',
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
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    # print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes-1) +
    #       ' adversarial examples')
    #
    # # Keep track of success (adversarial example classified in target)
    # results = np.zeros((nb_classes, source_samples), dtype='i')
    #
    # # Rate of perturbed features for each test set example and target class
    # perturbations = np.zeros((nb_classes, source_samples), dtype='f')
    #
    # # Initialize our array for grid visualization
    # grid_shape = (nb_classes, nb_classes, img_rows, img_cols, channels)
    # grid_viz_data = np.zeros(grid_shape, dtype='f')
    #
    # # Instantiate a SaliencyMapMethod attack object
    # jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    # jsma_params = {'theta': 1., 'gamma': 0.1,
    #                'clip_min': 0., 'clip_max': 1.,
    #                'y_target': None}
    #
    # figure = None
    #
    # # Loop over the samples we want to perturb into adversarial examples
    # adv_count = 0
    # for sample_ind in xrange(0, source_samples):
    #     print('--------------------------------------')
    #     print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
    #     sample = X_test[sample_ind:(sample_ind+1)]
    #
    #     label = model_argmax(sess, x, preds, sample)
    #
    #     # We want to find an adversarial example for each possible target class
    #     # (i.e. all classes that differ from the label given in the dataset)
    #     current_class = int(np.argmax(Y_test[sample_ind]))
    #     target_classes = other_classes(nb_classes, current_class)
    #
    #     # For the grid visualization, keep original images along the diagonal
    #     grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
    #         sample, (img_rows, img_cols, channels))
    #
    #     # Loop over all target classes
    #     store_path = './cifar10_adv_jsma'
    #     if not os.path.exists(store_path):
    #         os.makedirs(store_path)
    #     for target in target_classes:
    #         print('Generating adv. example for target class %i' % target)
    #
    #         # This call runs the Jacobian-based saliency map approach
    #         one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
    #         one_hot_target[0, target] = 1
    #         jsma_params['y_target'] = one_hot_target
    #         adv_x = jsma.generate_np(sample, **jsma_params)
    #
    #         # Check if success was achieved
    #         new_class_label = model_argmax(sess, x, preds, adv_x) # Predicted class of the generated adversary
    #         res = int(new_class_label != current_class)
    #
    #
    #         if res==1:
    #             adv_count += 1
    #             adv_img_deprocessed = deprocess_image(adv_x)
    #             imsave(store_path + '/adv_' + str(adv_count) + '_' + str(current_class) + '_' + str(new_class_label) + '.png', adv_img_deprocessed)
    #
    #         # # Computer number of modified features
    #         # adv_x_reshape = adv_x.reshape(-1)
    #         # test_in_reshape = X_test[sample_ind].reshape(-1)
    #         # nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
    #         # percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]
    #         #
    #         # # Display the original and adversarial images side-by-side
    #         #
    #         #
    #         # if viz_enabled:
    #         #     figure = pair_visual(
    #         #         np.reshape(sample, (img_rows, img_cols)),
    #         #         np.reshape(adv_x, (img_rows, img_cols)), figure)
    #         #
    #         # # Add our adversarial example to our grid data
    #         # grid_viz_data[target, current_class, :, :, :] = np.reshape(
    #         #     adv_x, (img_rows, img_cols, channels))
    #         #
    #         # # Update the arrays for later analysis
    #         # results[target, sample_ind] = res
    #         # perturbations[target, sample_ind] = percent_perturb
    #
    # print('--------------------------------------')
    #
    # # Generate random matution matrix for mutations
    # home = expanduser("~")
    # [image_list, real_labels, predicted_labels] = utils.get_data_mutation_test(home + '/cleverhans/cleverhans_tutorials/cifar10_adv_jsma')
    # img_rows = 32
    # img_cols = 32
    # seed_number = len(predicted_labels)
    # mutation_number = 1000
    #
    # mutation_test = MutationTest(img_rows, img_cols, seed_number, mutation_number)
    # mutations = []
    # for i in range(mutation_number):
    #     mutation = mutation_test.mutation_matrix()
    #     mutations.append(mutation)
    #
    # for step_size in [1,5,10]:
    #
    #     label_change_numbers = []
    #     # Iterate over all the test data
    #     for i in range(len(predicted_labels)):
    #         ori_img = preprocess_image(image_list[i].astype('float64'))
    #         orig_label = predicted_labels[i]
    #
    #         label_changes = 0
    #         for j in range(mutation_test.mutation_number):
    #             img = ori_img.copy()
    #             add_mutation = mutations[j][0]
    #             mu_img = img + add_mutation * step_size
    #             # print("----------------------------------------")
    #             # print(add_mutation)
    #
    #             # Predict the label for the mutation
    #             mu_img = np.expand_dims(mu_img, 0)
    #             # print(mu_img.shape)
    #             # Define input placeholder
    #             # input_x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    #
    #             mu_label = model_argmax(sess, x, preds, mu_img)
    #             # print('Predicted label: ', mu_label)
    #
    #             if mu_label != orig_label:
    #                 label_changes += 1
    #
    #         label_change_numbers.append(label_changes)
    #
    #     print('Number of label changes for step size: ' + str(step_size)+ ', '+ str(label_change_numbers))

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
    cifar10_tutorial_jsma(trained = FLAGS.trained,
                           viz_enabled=FLAGS.viz_enabled,
                           nb_epochs=FLAGS.nb_epochs,
                           batch_size=FLAGS.batch_size,
                           nb_classes=FLAGS.nb_classes,
                           source_samples=FLAGS.source_samples,
                           learning_rate=FLAGS.learning_rate)





if __name__ == '__main__':
    flags.DEFINE_boolean('trained', False, 'The model is already trained.')  #default:False
    flags.DEFINE_boolean('viz_enabled', True, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 60, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 100, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate for training')

    tf.app.run()
