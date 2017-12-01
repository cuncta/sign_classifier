Datasets: in data folder, in h5 format, already split:

Train set: 1080 pictures (64 by 64 pixels) of signs representing 
	numbers from 0 to 5 (180 pictures per number).
Test set: 120 pictures (64 by 64 pixels) of signs representing 
	numbers from 0 to 5 (20 pictures per number).




sign_classifier_0.0.py

model: 2 layer NN

LINEAR/RELU - LINEAR/RELU - LINEAR/SOFTMAX

input: 64(pixel)*64(pixel)*3(rgb) = 12288, linear/Relu 
1st: 25 neurons, linear/Relu
2nd: 12 neurons, linear/Relu
output: 6, linear/softmax


performance:

accuracy train: 0.999074
accuracy test: 0.725
training time: 330 sec

TO DO:
since there is a clear variance problem, I should try to add regularization.


conv_sign_classifier_0.0.py

model: convolutional NN

CONV2D\RELU\MAXPOOL -> CONV2D\RELU\MAXPOOL -> FLATTEN -> FC -> LINEAR/SOFTMAX

input: 64(pixel)*64(pixel)*3(rgb) = 12288, linear/Relu 
1st: conv-Relu-MaxPool
2nd: conv-Relu-MaxPool
3rd: fully connected
output: 6, linear/softmax


performance:

accuracy train: 0.959259
accuracy test: 0.875
training time: 316 sec

TO DO:
-the model should perform better on train set
-there is a clear variance problem, I should try to add regularization.