import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import *
from PIL import Image
from scipy import ndimage
import time


start = time.time()

#SIGNS Dataset

#
# Train set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
# Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).
print('loading SIGN datasets')
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()


# Example of a picture
#~ index = 0
#~ plt.imshow(X_train_orig[index])
#~ print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = one_hot_encoding(Y_train_orig, 6)
Y_test = one_hot_encoding(Y_test_orig, 6)


print ("train set size" + str(X_train.shape[1]))
print ("test set size" + str(X_test.shape[1]))
print ("X_train shape " + str(X_train.shape))
print ("Y_train shape " + str(Y_train.shape))
print ("X_test shape " + str(X_test.shape))
print ("Y_test shape " + str(Y_test.shape))
print("START TRAINING")

ops.reset_default_graph()                  # Clears the default graph stack and resets the global default graph. To  rerun the model without overwriting tf variables
tf.set_random_seed(1)                      # 1 to keep consistent results
(n_x, m) = X_train.shape                 # (n_x: input size, m : number of examples in the train set)
n_y = Y_train.shape[0]                    # n_y : output size
costs = []                                       # To keep track of the cost
epoch_n = []
learning_rate = 0.0001
num_epochs = 1500
minibatch_size = 32 
seed=3


### CREATE PLACEHOLDERS
X = tf.placeholder(dtype = tf.float32, shape = [n_x, None], name = "X")
Y = tf.placeholder(dtype = tf.float32, shape = [n_y, None], name = "Y")

###INITIALIZE PARAMETERS
W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b3 = tf.get_variable("b4", [6,1], initializer = tf.zeros_initializer())

parameters = {"W1": W1, "b1": b1, "W2": W2,"b2": b2, "W3": W3, "b3": b3}

### FORWARD PROPAGATION
#LINEAR/RELU - LINEAR/RELU - LINEAR/SOFTMAX
Z1 = tf.add(tf.matmul(W1,X),b1)                        
A1 = tf.nn.relu(Z1)                                             
Z2 = tf.add(tf.matmul(W2,A1),b2)                      
A2 = tf.nn.relu(Z2)                                             
Z3 = tf.add(tf.matmul(W3,A2),b3)     

###CALCULATE COST
logits = tf.transpose(Z3)
labels = tf.transpose(Y)    
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

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
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size,seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # Run the session: it executes the "optimizer" and the "cost" on a minibatch contained in the feedict (X,Y).
                _ , minibatch_cost = sess.run([optimizer,cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                epoch_n.append(epoch)
                
        # plot the cost
        plt.plot(epoch_n,costs)
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.title("Learning rate =" + str(learning_rate))
        plt.savefig('intermediate/v_0.0_parameters_num_epochs_'+str(num_epochs)+'.pdf')

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
        np.save('intermediate/v_0.0_parameters_num_epochs_'+str(num_epochs)+'.npy', parameters) 



end = time.time()
print('END TRAINING')
print('time:', end - start, 'sec')
plt.show()