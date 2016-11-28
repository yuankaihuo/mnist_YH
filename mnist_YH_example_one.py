
import mnist_cnn as cnn
import numpy as np
import sys

#addnoise = int(sys.argv[1])
#print("addnoise = %d" % addnoise)
#
#randomlabel= int(sys.argv[2])
#print("randomlabel = %d" % randomlabel)
#
#mu = int(sys.argv[3])
#print("mu = %d" % mu )
#
#std = int(sys.argv[4])
#print("std = %d" % std )
#
#randomrate = float(sys.argv[5])
#print("randomrate = %f" % randomrate )

addnoise = 1;
randomlabel = 0;
mu = 0;
std = 128;
randomrate = 0;


from tensorflow.examples.tutorials.mnist import input_data
print ("=== PACKAGES LOADED ===")

mnist = input_data.read_data_sets('data/', one_hot=True)
print ("=== MNIST READY ===")

#Specify how many epochs you want during training 
training_epochs = 15
weights,biases = cnn.variable_init()
cost,optm,accr,sess,_corr = cnn.init_graph(weights,biases)


expriments = {'add_gaussian_noise':addnoise , 'add_label_noise':randomlabel, 'mu':mu, 'std':std, 'label_random_rate':randomrate }
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriments)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs,cost,optm,accr,sess)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)