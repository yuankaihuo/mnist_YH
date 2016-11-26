# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 20:24:45 2016

@author: Yuankai Huo @ Vanderbilt University
"""

import mnist_cnn as cnn
from tensorflow.examples.tutorials.mnist import input_data
print ("=== PACKAGES LOADED ===")

mnist = input_data.read_data_sets('data/', one_hot=True)
print ("=== MNIST READY ===")

training_epochs = 10

expriment1 = {'add_gaussian_noise':0 , 'add_label_noise':0}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment1)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)

expriment2 = {'add_gaussian_noise':1 , 'add_label_noise':0, 'mu':0, 'std':8}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment2)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)

expriment3 = {'add_gaussian_noise':1 , 'add_label_noise':0, 'mu':0, 'std':32}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment3)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)

#input_r = cnn.get_tf_out(mnist.train.images,'input_r')
#cnn.plot_input(input_r,0)
#
#expriment4 = {'add_gaussian_noise':1 , 'add_label_noise':0, 'mu':0, 'std':8}
#trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment4)
##sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs)
#input_r = cnn.get_tf_out(trainimg,'input_r')
#cnn.plot_input(input_r,0)
#
#expriment4 = {'add_gaussian_noise':1 , 'add_label_noise':0, 'mu':0, 'std':32}
#trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment4)
##sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs)
#input_r = cnn.get_tf_out(trainimg,'input_r')
#cnn.plot_input(input_r,0)
#
#expriment4 = {'add_gaussian_noise':1 , 'add_label_noise':0, 'mu':0, 'std':128}
#trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment4)
##sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs)
#input_r = cnn.get_tf_out(trainimg,'input_r')
#cnn.plot_input(input_r,0)




#test_acc = cnn.test_net(testimg,testlabel,sess,accr)



expriment5 = {'add_gaussian_noise':0, 'add_label_noise':1, 'label_random_rate':0.05}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment5)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)

expriment6 = {'add_gaussian_noise':0, 'add_label_noise':1, 'label_random_rate':0.15}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment6)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)

expriment7 = {'add_gaussian_noise':0, 'add_label_noise':1, 'label_random_rate':0.5}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment7)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)