import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from sign_class_lib import *
from PIL import Image
from scipy import ndimage
import time


start = time.time()

#SIGNS Dataset

#
# - **Train set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
# - **Test set**: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).
print('loading SIGN datasets')
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()


# Example of a picture
#~ index = 0
#~ plt.imshow(X_train_orig[index])
#~ print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


X_train = X_train_orig/255.
X_test = X_test_orig/255.
# Convert training and test labels to one hot matrices
Y_train = one_hot_encoding(Y_train_orig, 6).T
Y_test = one_hot_encoding(Y_test_orig, 6).T


print ("train set size" + str(X_train.shape[0]))
print ("test set size" + str(X_test.shape[0]))
print ("X_train shape " + str(X_train.shape))
print ("Y_train shape " + str(Y_train.shape))
print ("X_test shape " + str(X_test.shape))
print ("Y_test shape " + str(Y_test.shape))
print("START TRAINING")

ops.reset_default_graph()                         # Clears the default graph stack and resets the global default graph. To  rerun the model without overwriting tf variables
tf.set_random_seed(1)                             # 1 to keep consistent results
(m, n_H0, n_W0, n_C0) = X_train.shape   # (n_x: input size, m : number of examples in the train set)
n_y = Y_train.shape[1]                            # n_y : output size
costs = []                                               # To keep track of the cost
epoch_n = []
learning_rate = 0.001
num_epochs = 200
minibatch_size = 64 
seed=3
conv_layers = {}



### CREATE PLACEHOLDERS
X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0]) 
Y = tf.placeholder(tf.float32,[None, n_y])

###INITIALIZE PARAMETERS
W1 = tf.get_variable("W1", [4,4,3,8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
W2 = tf.get_variable("W2", [2,2,8,16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

parameters = {"W1": W1, "W2": W2}


### FORWARD PROPAGATION
#CONV2D\RELU -> MAXPOOL -> CONV2D\RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
A1 = tf.nn.relu(Z1)
P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding='SAME')
Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
A2 = tf.nn.relu(Z2)
P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding='SAME')
P = tf.contrib.layers.flatten(P2)
Z3 = tf.contrib.layers.fully_connected(P, 6, activation_fn=None)     

###CALCULATE COST  
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))

### BACKPROPAGATION WITH ADAM
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

### INITIALIZE VARIABLES
init = tf.global_variables_initializer()

### START SESSION
seed=3
with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches_conv(X_train, Y_train, minibatch_size,seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # Run the session: it executes the "optimizer" and the "cost" on a minibatch contained in the feedict (X,Y).
                _ , minibatch_cost = sess.run([optimizer,cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if epoch % 1 == 0:
                costs.append(epoch_cost)
                epoch_n.append(epoch)
                
        # plot the cost
        plt.plot(epoch_n,costs)
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.title("Learning rate =" + str(learning_rate))
        plt.savefig('intermediate/parameters_num_epochs_'+str(num_epochs)+'.pdf')

        #plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        np.save('intermediate/parameters_num_epochs_'+str(num_epochs)+'.npy', parameters) 



end = time.time()
print('END TRAINING')
print('time:', end - start, 'sec')
plt.show()