
import numpy as np
from scipy.misc import imresize
from scipy import ndimage
import PIL
from sign_class_lib import predict
import matplotlib.pyplot as plt
import tensorflow as tf
import math


parameters = np.load('intermediate/parameters_num_epochs_1501.npy').item()
# test with an image not from database
test_image = "0.jpg"
#~ test_image = "1.jpg"
#~ test_image = "2.jpg"
#~ test_image = "3.jpg"
#~ test_image = "4.jpg"
#~ test_image = "5.jpg"

# Preprocess the image to fit the algorithm.
fname = "images/" + test_image
image = np.array(ndimage.imread(fname, flatten=False))
test_image = imresize(image, size=(64,64)).reshape((1, 64*64*3)).T

#predict
#~ my_image_prediction = predict(my_image, parameters)
W1 = tf.convert_to_tensor(parameters["W1"])
b1 = tf.convert_to_tensor(parameters["b1"])
W2 = tf.convert_to_tensor(parameters["W2"])
b2 = tf.convert_to_tensor(parameters["b2"])
W3 = tf.convert_to_tensor(parameters["W3"])
b3 = tf.convert_to_tensor(parameters["b3"])

params = {"W1": W1,
	"b1": b1,
	"W2": W2,
	"b2": b2,
	"W3": W3,
	"b3": b3}

x = tf.placeholder("float", [12288, 1])

#~ z3 = forward_propagation(x, params)
Z1 = tf.add(tf.matmul(W1, x), b1)    
A1 = tf.nn.relu(Z1)                           
Z2 = tf.add(tf.matmul(W2, A1), b2)  
A2 = tf.nn.relu(Z2)                           
Z3 = tf.add(tf.matmul(W3, A2), b3)
p = tf.argmax(Z3)

sess = tf.Session()
prediction = sess.run(p, feed_dict = {x: test_image})


plt.imshow(image)
print("prediction: y = " + str(np.squeeze(prediction)))
plt.show()