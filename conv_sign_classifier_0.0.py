
# coding: utf-8

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from sign_class_lib import *
import time



start = time.time()


# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
#~ index = 6
#~ plt.imshow(X_train_orig[index])
#~ print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = one_hot_encoding(Y_train_orig, 6).T
Y_test = one_hot_encoding(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}


ops.reset_default_graph()                         # Clears the default graph stack and resets the global default graph. To  rerun the model without overwriting tf variables
tf.set_random_seed(1)                             # to keep results consistent
seed = 3                                                  # to keep results consistent (numpy seed)
(m, n_H0, n_W0, n_C0) = X_train.shape             
n_y = Y_train.shape[1]                            
costs = []   
epoch_n = []
learning_rate = 0.001
num_epochs = 200
minibatch_size = 64


### CREATE PLACEHOLDERS
X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0]) 
Y = tf.placeholder(tf.float32,[None, n_y])
    
###INITIALIZE PARAMETERS    
tf.set_random_seed(1)                          
        
W1 = tf.get_variable("W1", [4,4,3,8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
W2 = tf.get_variable("W2", [2,2,8,16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

parameters = {"W1": W1, "W2": W2}

### FORWARD PROPAGATION
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
     
# Start the session to compute the tensorflow graph
with tf.Session() as sess:
    
    # Run the initialization
    sess.run(init)
    
    # Do the training loop
    for epoch in range(num_epochs):

        epoch_cost = 0.
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches_conv(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            # IMPORTANT: The line that runs the graph on a minibatch.
            # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
            _ , minibatch_cost = sess.run([optimizer,cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
            
            epoch_cost += minibatch_cost / num_minibatches
            

        # Print the cost every epoch
        if epoch % 5 == 0:
            print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
        if epoch % 1 == 0:
            costs.append(epoch_cost)
            epoch_n.append(epoch)
    
    
    # plot the cost
    plt.plot(epoch_n,costs)
    plt.ylabel('cost')
    plt.xlabel('epoch')
    plt.title("Learning rate =" + str(learning_rate))
    #~ plt.savefig('intermediate/conv_cost_num_epochs_'+str(num_epochs)+'.pdf')

    # lets save the parameters in a variable
    parameters = sess.run(parameters)
    print ("Parameters have been trained!")
    
    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y, 1))
    
    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
    #~ np.save('intermediate/conv_parameters_num_epochs_'+str(num_epochs)+'.npy', parameters) 


end = time.time()
print('END TRAINING')
print('time:', end - start, 'sec')
plt.show()



