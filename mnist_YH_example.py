# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 20:24:45 2016

@author: Yuankai Huo @ Vanderbilt University
"""

import mnist_cnn as cnn
from tensorflow.examples.tutorials.mnist import input_data
print ("=== PACKAGES LOADED ===")

#load mnist dataset
mnist = input_data.read_data_sets('data/', one_hot=True)
print ("=== MNIST READY ===")

#Specify how many epochs you want during training 
training_epochs = 1
weights,biases = cnn.variable_init()
cost,optm,accr,sess,_corr = cnn.init_graph(weights,biases)

#cnn.show_trainimg(mnist.train.images,sess,weights,biases)


#setup the first experiement, using cnn
expriment1 = {'add_gaussian_noise':0 , 'add_label_noise':0}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment1)
cnn.show_trainimg(trainimg,sess,weights,biases)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs,cost,optm,accr,sess)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)
test_acc_each_class = cnn.test_net_each_class(testimg,testlabel,sess,_corr)

expriment2 = {'add_gaussian_noise':1 , 'add_label_noise':0, 'mu':0, 'std':8}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment2)
cnn.show_trainimg(trainimg,sess,weights,biases)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs,cost,optm,accr,sess)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)

expriment3 = {'add_gaussian_noise':1 , 'add_label_noise':0, 'mu':0, 'std':32}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment3)
cnn.show_trainimg(trainimg,sess,weights,biases)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs,cost,optm,accr,sess)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)

expriment4 = {'add_gaussian_noise':1 , 'add_label_noise':0, 'mu':0, 'std':128}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment4)
cnn.show_trainimg(trainimg,sess,weights,biases)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs,cost,optm,accr,sess)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)



expriment5 = {'add_gaussian_noise':0, 'add_label_noise':1, 'label_random_rate':0.05}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment5)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs,cost,optm,accr,sess)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)

expriment6 = {'add_gaussian_noise':0, 'add_label_noise':1, 'label_random_rate':0.15}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment6)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs,cost,optm,accr,sess)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)

expriment7 = {'add_gaussian_noise':0, 'add_label_noise':1, 'label_random_rate':0.5}
trainimg, trainlabel, testimg, testlabel, cnn_file_name = cnn.init_expr(mnist,expriment7)
sess,accr = cnn.train_net(trainimg,trainlabel,cnn_file_name,training_epochs,cost,optm,accr,sess)
test_acc = cnn.test_net(testimg,testlabel,sess,accr)