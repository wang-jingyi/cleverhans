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
import random
import math

import os
import sys
sys.path.append('/Users/pxzhang/Documents/SUTD/project/deepxplore')

from MNIST_mine import utils
from MNIST_mine.gen_mutation import MutationTest
# from utils import deprocess_mnist_image

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, model_argmax
from cleverhans.utils_keras import KerasModelWrapper, cnn_model
from cleverhans_tutorials.tutorial_models import make_basic_cnn

FLAGS = flags.FLAGS

def mnist_tutorial_jsma(trained = True, mutated = True, train_start=0, train_end=60000, test_start=0,
                        test_end=10000, viz_enabled=True, nb_epochs=6,
                        batch_size=128, nb_classes=10, source_samples=10,
                        learning_rate=0.001):
    """
    MNIST tutorial for the Jacobian-based saliency map approach (JSMA)
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

    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    # Define TF model graph
    model = make_basic_cnn()
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
        'train_dir': 'mnist_jsma/model',
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
    print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes-1) +
          ' adversarial examples')

    # Keep track of success (adversarial example classified in target)
    results = np.zeros((nb_classes, source_samples), dtype='i')

    # Rate of perturbed features for each test set example and target class
    perturbations = np.zeros((nb_classes, source_samples), dtype='f')

    # Initialize our array for grid visualization
    grid_shape = (nb_classes, nb_classes, img_rows, img_cols, channels)
    grid_viz_data = np.zeros(grid_shape, dtype='f')

    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    figure = None

    # Loop over the samples we want to perturb into adversarial examples
    count = 0
    for sample_ind in xrange(0, source_samples):
        print('--------------------------------------')
        print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
        sample = X_test[sample_ind:(sample_ind+1)]

        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(Y_test[sample_ind]))
        target_classes = other_classes(nb_classes, current_class)

        # For the grid visualization, keep original images along the diagonal
        grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
            sample, (img_rows, img_cols, channels))

        # Loop over all target classes
        store_path = './mnist_jsma/adv_jsma'
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        for target in target_classes:
            count = count + 1
            print('Generating adv. example for target class %i' % target)

            # This call runs the Jacobian-based saliency map approach
            one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
            one_hot_target[0, target] = 1
            jsma_params['y_target'] = one_hot_target
            adv_x = jsma.generate_np(sample, **jsma_params)

            # Check if success was achieved
            new_class_label = model_argmax(sess, x, preds, adv_x) # Predicted class of the generated adversary
            res = int(new_class_label == target)

            # if res==1:
            adv_img_deprocessed = utils.deprocess_image(adv_x)
            imsave(store_path + '/adv_' + str(count) + '_' + str(current_class) + '_' + str(new_class_label) + '_.png', adv_img_deprocessed)

            if count == 500:
                break

        if count == 500:
            break

    print('--------------------------------------')

    # Generate random matution matrix for mutations
    path = './mnist_jsma'
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
    for step_size in [1,5,10]:

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

                mu_label = model_argmax(sess, x, preds, mu_img)
                # print('Predicted label: ', mu_label)

                if mu_label != int(orig_label):
                    label_changes += 1

            label_change_numbers.append(label_changes)
            store_string = store_string + image_files[i] + "," + str(step_size) + "," + str(label_changes) + "\n"

        label_change_numbers = np.asarray(label_change_numbers)
        adv_average = round(np.mean(label_change_numbers), 2)
        adv_std = np.std(label_change_numbers)
        adv_95ci = round(1.96 * adv_std / math.sqrt(len(label_change_numbers)), 2)
        result = result + 'adv,' + str(step_size) + ',' + str(adv_average) + ',' + str(round(adv_std, 2)) + ',' + str(adv_95ci) + '\n'

        # print('Number of label changes for step size: ' + str(step_size)+ ', '+ str(label_change_numbers))

    with open(path + "/adv_result.csv", "w") as f:
        f.write(store_string)

    store_path = './mnist_jsma/ori_jsma'
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
            orig_label = model_argmax(sess, x, preds, np.asarray([image_list[i]]))

            label_changes = 0
            for j in range(mutation_number):
                img = ori_img.copy()
                add_mutation = mutations[j][0]
                mu_img = img + add_mutation * step_size

                # Predict the label for the mutation
                mu_img = np.expand_dims(mu_img, 0)

                mu_label = model_argmax(sess, x, preds, mu_img)
                # print('Predicted label: ', mu_label)

                if mu_label != int(orig_label):
                    label_changes += 1

            label_change_numbers.append(label_changes)
            store_string = store_string + str(i) + "," + str(step_size) + "," + str(label_changes) + "\n"

        label_change_numbers = np.asarray(label_change_numbers)
        adv_average = round(np.mean(label_change_numbers), 2)
        adv_std = np.std(label_change_numbers)
        adv_95ci = round(1.96 * adv_std / math.sqrt(len(label_change_numbers)), 2)
        result = result + 'ori,' + str(step_size) + ',' + str(adv_average) + ',' + str(round(adv_std, 2)) + ',' + str(adv_95ci) + '\n'
        # print('Number of label changes for step size: ' + str(step_size)+ ', '+ str(label_change_numbers))

    with open(path + "/ori_result.csv", "w") as f:
        f.write(store_string)

    with open(path + "/result.csv", "w") as f:
        f.write(result)

    # Close TF session
    sess.close()
    print('Finish generating adversaries.')

    return report


def main(argv=None):
    mnist_tutorial_jsma(trained = FLAGS.trained,
                        mutated = FLAGS.mutated,
                        viz_enabled=FLAGS.viz_enabled,
                        nb_epochs=FLAGS.nb_epochs,
                        batch_size=FLAGS.batch_size,
                        nb_classes=FLAGS.nb_classes,
                        source_samples=FLAGS.source_samples,
                        learning_rate=FLAGS.learning_rate)





if __name__ == '__main__':
    flags.DEFINE_boolean('trained', False, 'The model is already trained.')  #default:False
    flags.DEFINE_boolean('mutated', False, 'The mutation list is already generate.')  # default:False
    flags.DEFINE_boolean('viz_enabled', True, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 60, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    tf.app.run()
