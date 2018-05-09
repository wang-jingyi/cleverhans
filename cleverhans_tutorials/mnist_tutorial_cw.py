"""
This tutorial shows how to generate adversarial examples
using C&W attack in white-box setting.
The original paper can be found at:
https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
from scipy.misc import imsave
import random
import math

import logging
import os


from MNIST_mine import utils
from MNIST_mine.gen_mutation import MutationTest

from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load, model_argmax
from cleverhans_tutorials.tutorial_models import make_basic_cnn

FLAGS = flags.FLAGS


def mnist_tutorial_cw(trained = True, mutated = True, train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=True, nb_epochs=6,
                      batch_size=128, nb_classes=10, source_samples=10,
                      learning_rate=0.001, attack_iterations=100,
                      model_path=os.path.join("models", "mnist"),
                      targeted=True):
    """
    MNIST tutorial for Carlini and Wagner's attack
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
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
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

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Define TF model graph
    model = make_basic_cnn()
    preds = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        # 'train_dir': os.path.join(*os.path.split(model_path)[:-1]),
        # 'filename': os.path.split(model_path)[-1]
        'train_dir': 'mnist_cw/model',
        'filename': 'cw.model'
    }

    rng = np.random.RandomState([2017, 8, 30])
    # check if we've trained before, and if we have, use that pre-trained model
    if trained:
        saver = tf.train.Saver()
        saver.restore(
            sess, os.path.join(
                train_params['train_dir'], train_params['filename']))
    else:
        model_train(sess, x, y, preds, X_train, Y_train, args=train_params,
                    rng=rng,save=True)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
    ###########################################################################
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
            (source_samples * nb_classes, img_rows, img_cols, 1))
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

    store_path = './mnist_cw/adv_cw'
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    new_class_labels = model_argmax(sess, x, preds, adv)

    for i in range(len(adv)):
    #      Predicted class of the generated adversary
        res = int(new_class_labels[i] == targets[i])
        # if res == 1:
        adv_img_deprocessed = utils.deprocess_image(np.asarray([adv[i]]))
        imsave(store_path + '/adv_' + str(i) + '_' + str(np.argmax(label[int(i/10)][int(i%10)],axis=0)) + '_' + str(new_class_labels[i]) + '_.png',
               adv_img_deprocessed)

    print('--------------------------------------')

    #Compute the number of different labels for mutation of successful adversarial examples
    path = './mnist_cw'
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
                # print(mu_img.shape)
                # Define input placeholder
                # input_x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))

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
        result = result + 'adv,' + str(step_size) + ',' + str(adv_average) + ',' + str(round(adv_std, 2)) + ',' + str(
            adv_95ci) + '\n'

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
                # print(mu_img.shape)
                # Define input placeholder
                # input_x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))

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
        result = result + 'ori,' + str(step_size) + ',' + str(adv_average) + ',' + str(round(adv_std, 2)) + ',' + str(
            adv_95ci) + '\n'

        # print('Number of label changes for step size: ' + str(step_size)+ ', '+ str(label_change_numbers))

    with open(path + "/ori_result.csv", "w") as f:
        f.write(store_string)

    with open(path + "/result.csv", "w") as f:
        f.write(result)

    # Close TF session
    sess.close()

    return report


def main(argv=None):
    mnist_tutorial_cw(trained = FLAGS.trained,
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
    flags.DEFINE_boolean('trained', False, 'The model is already trained.')  # default:False
    flags.DEFINE_boolean('mutated', False, 'The mutation list is already generate.')  # default:False
    flags.DEFINE_boolean('viz_enabled', True, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 50, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                        'Path to save or load the model file')
    flags.DEFINE_integer('attack_iterations', 100,
                         'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', True,
                         'Run the tutorial in targeted mode?')

    tf.app.run()
