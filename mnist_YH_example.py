# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 20:24:45 2016

@author: Yuankai Huo @ Vanderbilt University

Code and CNN Reference 
https://github.com/sjchoi86/Tensorflow-101
"""

# import the cnn code and mnist dataset
import mnist_cnn as cnn
from tensorflow.examples.tutorials.mnist import input_data

#==============================================================================
#Set up experiement
training_epochs = 1;    # Specify how many epochs during training
show_training_image = False; # True means show 100 training images

#==============================================================================
#Define experiements
              # training on raw mnist images
expriments = ({'add_gaussian_noise':0 , 'add_label_noise':0},  
              # training on images with Gaussian noise. (different mean and standard deviation)
              {'add_gaussian_noise':1 , 'add_label_noise':0, 'mu':0, 'std':8}, 
              {'add_gaussian_noise':1 , 'add_label_noise':0, 'mu':0, 'std':32},
              {'add_gaussian_noise':1 , 'add_label_noise':0, 'mu':0, 'std':128},
                # training on images with label noise.  (different percentage of randomize)
              {'add_gaussian_noise':0, 'add_label_noise':1, 'label_random_rate':0.05},
              {'add_gaussian_noise':0, 'add_label_noise':1, 'label_random_rate':0.05},
              {'add_gaussian_noise':0, 'add_label_noise':1, 'label_random_rate':0.5});
              

#==============================================================================
#load mnist dataset
mnist = input_data.read_data_sets('data/', one_hot=True)
#initialize the CNN network
weights,biases = cnn.variable_init()
#initialize the graph
cost,optm,accr,sess,_corr = cnn.init_graph(weights,biases)

#==============================================================================
#run training and testing\
n = 1;
for experiment in expriments:
    print("*** Start experiment %d*********************************************\n" % n)
    #initialize the training and testing data
    trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,experiment)
    #show the first 100 training images
    if show_training_image:
    	cnn.show_trainimg(trainimg,sess,weights,biases)
    #train CNN
    sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs,cost,optm,accr,sess)
    #test CNN
    test_acc = cnn.test_net(testimg,testlabel,sess,accr)
    test_acc_each_class = cnn.test_net_each_class(testimg,testlabel,sess,_corr)
    print("*** Finish experiment %d********************************************\n" % n)
    n += 1
