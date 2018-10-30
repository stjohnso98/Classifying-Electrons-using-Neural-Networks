
""" Build an Image Dataset in TensorFlow.
For this example, you need to make your own set of images (JPEG).
We will show 2 different ways to build that dataset:
- From a root folder, that will have a sub-folder containing images for each class
    ```
    ROOT_FOLDER
       |-------- SUBFOLDER (CLASS 0)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
       |             
       |-------- SUBFOLDER (CLASS 1)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
    ```
- From a plain text file, that will list all images with their class ID:
    ```
    /path/to/image/1.jpg CLASS_ID
    /path/to/image/2.jpg CLASS_ID
    /path/to/image/3.jpg CLASS_ID
    /path/to/image/4.jpg CLASS_ID
    etc...
    ```
Below, there are some parameters that you need to change (Marked 'CHANGE HERE'), 
such as the dataset path.

After creating image dataset seperately for training and testing, training and testing is done for the respective dataset and total accuracy of the model is printed
Author: Steenu Johnson
Project: https://github.com/stjohnso98/Tensorflow
"""
from __future__ import print_function

import tensorflow as tf
import os

# Dataset Parameters - CHANGE HERE
MODE = 'folder' # or 'file', if you choose a plain text file (see above).
TRAIN_DATASET_PATH = '/home/dell/dataset/Medium_herwig/train_ele' # the dataset file or root folder path.
TEST_DATASET_PATH = '/home/dell/dataset/Medium_herwig/test_ele'

# Image Parameters
N_CLASSES = 2 # CHANGE HERE, total number of classes
IMG_HEIGHT = 64 # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 64 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale


# Reading the dataset
# 2 modes: 'file' or 'folder'
def read_images(dataset_path, mode, batch_size):
    imagepaths, labels = list(), list()
    if mode == 'file':
        # Read dataset file
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        # An ID will be affected to each sub-folders by alphabetical order
        label = 0
        # List the directory
        try:  # Python 2
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:  # Python 3
            classes = sorted(os.walk(dataset_path).__next__()[1])
        # List each sub-directory (the classes)
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            try:  # Python 2
                walk = os.walk(c_dir).next()
            except Exception:  # Python 3
                walk = os.walk(c_dir).__next__()
            # Add each image to the training set
            for sample in walk[2]:
                # Only keeps jpeg images
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
            label += 1
    else:
        raise Exception("Unknown mode.")

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    return X, Y

# -----------------------------------------------
# THIS IS A CLASSIC CNN (see examples, section 3)
# -----------------------------------------------
# Note that a few elements have changed (usage of queues).

# Parameters
learning_rate = 0.001
total_img = 7524 # Total Number of images
batch_size = 132 # Batch size for training
if total_img<=132 :
    batch_size_test = int(total_img) # Batch size for testing
else:
    batch_size_test = 132
epoch_steps= total_img/batch_size
num_epoch = 5 # Epoch is one complete presentation of data
num_steps = int(num_epoch * epoch_steps)
num_steps_test = int(total_img/batch_size_test)
display_step = 1
dropout = 0.75 # Dropout, probability to keep units

# Build the data input
X_train, Y_train = read_images(TRAIN_DATASET_PATH, MODE, batch_size)
X_train_test, Y_train_test = read_images(TRAIN_DATASET_PATH, MODE, batch_size_test) # A copy of training data for testing once the training is complete
X_test, Y_test = read_images(TEST_DATASET_PATH, MODE, batch_size_test)

# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(X_train, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights
logits_test = conv_net(X_train, N_CLASSES, dropout, reuse=True, is_training=False)
logits_train_test = conv_net(X_train_test,N_CLASSES, dropout, reuse=True, is_training=False)
# Create another graph for testing that reuse the same weights
logits_test_test = conv_net(X_test, N_CLASSES, dropout, reuse=True, is_training=False)
# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y_train))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y_train, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
correct_pred_train = tf.equal(tf.argmax(logits_train_test, 1), tf.cast(Y_train_test, tf.int64),name="correct_pred_train")
accuracy_train = tf.reduce_mean(tf.cast(correct_pred_train, tf.float32),name="train_accuracy")

correct_pred_test = tf.equal(tf.argmax(logits_test_test, 1), tf.cast(Y_test, tf.int64),name="correct_pred_test")
accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32),name="test_accuracy")

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Saver object
saver = tf.train.Saver()
sum_test = 0
sum_train = 0
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    # Start the data queue
    tf.train.start_queue_runners()

    # Training cycle
    for step in range(1, num_steps+1):

        if step % display_step == 0:
            # Run optimization and calculate batch loss and accuracy
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)

    print("Optimization Finished!")
    for step in range(1, num_steps_test+1):
        train_acc = sess.run(accuracy_train)
        test_acc = sess.run(accuracy_test)
        sum_train = float(sum_train) + float(train_acc)
        sum_test = float(sum_test) + float(test_acc)
        print("Step " + str(step) + ", Training Accuracy= " + \
                  "{:.4f}".format(train_acc) + ", Testing Accuracy= "+"{:.4f}".format(test_acc))
    average_train=float(sum_train/num_steps_test)
    average_test=float(sum_test/num_steps_test)
    print("Training Accuracy = "+ "{:.4f}".format(average_train) + ", Testing Accuracy = " + "{:.4f}".format(average_test))
    saver.save(sess, '/home/dell/my_tf_model')
    coord.request_stop()
    coord.join(threads)
    # Save your model
    #saver.save(sess, 'my_tf_model')
