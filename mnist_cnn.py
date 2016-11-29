# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 20:24:45 2016

@author: osboxes
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# initialize parameters
n_input      = 784
n_output     = 10
batch_size   = 100
display_step = 1
   
# initialize place holder
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# initialize the training and testing data as well as the intermediate file name
def init_expr(mnist,expr):
    #get data from mnist
    trainimg   = np.copy(mnist.train.images)
    trainlabel = np.copy(mnist.train.labels)
    testimg    = np.copy(mnist.test.images)
    testlabel  = np.copy(mnist.test.labels)
    
    # get expr parameters
    add_gaussian_noise = expr['add_gaussian_noise']
    add_label_noise = expr['add_label_noise']
    
    if (add_gaussian_noise==1):
        mu = expr['mu']
        std = expr['std']
        # add Gaussian noise to training images
        trainimg = add_noise(trainimg,mu,std)
        cnn_file_name = "mnist_gaussian_noise_mu%d_std%d.ckpt-" % (mu,std)
        print ("--- Add gaussian noise mu=%d std=%d ---" % (mu,std))
    elif (add_label_noise==1):
        label_random_rate = expr['label_random_rate']
        # randomize the training labels
        trainlabel = random_label(trainlabel,label_random_rate)
        cnn_file_name = "mnist_label_noise_%.2f.ckpt-" % (label_random_rate)
        print ("--- Randomize %.2f percent trainlabel  ---" % (label_random_rate*100))
    else:
        # using default training dataaset
        cnn_file_name = "mnist_cnn.ckpt-"
        
    return trainimg, trainlabel, testimg, testlabel, cnn_file_name

# reshape the 784x1 training image to 28x28 image
def get_images(training_set):
    """ Return a list containing the images from the MNIST data
    set. Each image is represented as a 2-d numpy array."""
    flattened_images = training_set
    return [np.reshape(f, (-1, 28)) for f in flattened_images]

# plot first 100 images in 10x10 format
def plot_images(trainimg):
    images = get_images(trainimg)
    #images = [image[3:25, 3:25] for image in images]
    rowImg = [];
    allImg = [];
    for x in range(10):
        for y in range(10):
            if y==0:
                rowImg = images[10*y+x]
            else:
                rowImg = np.concatenate((rowImg,images[10*y+x]),axis=1);
        if x==0:
            allImg = rowImg;
        else:
            allImg = np.concatenate((allImg,rowImg),axis=0);
    plt.matshow(allImg, cmap=plt.get_cmap('gray'))
    plt.axis('off')

# plot first 100 images in 10x10 format for multi-channel images
def plot_input(images,channel):
    rowImg = [];
    allImg = [];
    for x in range(10):
        for y in range(10):
            if y==0:
                rowImg = images[10*y+x,:,:,channel]
            else:
                rowImg = np.concatenate((rowImg,images[10*y+x,:,:,channel]),axis=1);
        if x==0:
            allImg = rowImg;
        else:
            allImg = np.concatenate((allImg,rowImg),axis=0);
    plt.matshow(allImg, cmap=plt.get_cmap('gray'))
    plt.axis('off')

# get intermediate images out from tensorflow graph
def get_tf_out(trainimg,out_name,sess,weights,biases):
    conv_out = conv_basic(x, weights, biases, keepratio)
    out_data = sess.run(conv_out[out_name], feed_dict={x: trainimg[0:100, :]})
    return out_data

# show training images in 10x10 format
def show_trainimg(trainimg,sess,weights,biases):
    input_r = get_tf_out(trainimg,'input_r',sess,weights,biases)
    plot_input(input_r,0)

# add Gaussian noise for different mean and standard deviation
def add_noise(trainimg,mu,std):
    for i in range(trainimg.shape[0]):
        if std==0:
            noise = mu
        else:
            noise = np.random.normal(mu,std,trainimg.shape[1]);
        trainimg_255 = np.multiply(trainimg[i], 255.0)
        trainimg_noised = trainimg_255+noise;
        trainimg_noised = np.clip(trainimg_noised,0.0,255.0)
        trainimg[i] = np.multiply(trainimg_noised, 1.0 / 255.0)
    return trainimg

# randomize labels for different percentage
def random_label(trainlabel,label_random_rate):
    train_size = trainlabel.shape[0]
    random_size =  int(train_size*label_random_rate)
    perm = np.arange(train_size)
    np.random.shuffle(perm)
    random_train_index = perm[0:random_size]
    random_train_label = np.random.randint(10,size=random_size)
    random_complete = 0
    for ind in random_train_index:
        label_num = random_train_label[random_complete]
        label_array = np.zeros(10)
        label_array[label_num] = 1.0
        trainlabel[ind] = label_array
        random_complete += 1
    return trainlabel
    
# initulize weights and biases
def variable_init():
    weights  = {
        'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.1)),
        'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
        'wd1': tf.Variable(tf.truncated_normal([7*7*128, 1024], stddev=0.1)),
        'wd2': tf.Variable(tf.truncated_normal([1024, n_output], stddev=0.1))
    }
    biases   = {
        'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
        'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
        'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
        'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
    }
    return weights,biases

# define the CNN
def conv_basic(_input, _w, _b, _keepratio):
    # INPUT
    _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
    # CONV LAYER 1
    _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    _mean, _var = tf.nn.moments(_conv1, [0, 1, 2])
    _conv1 = tf.nn.batch_normalization(_conv1, _mean, _var, 0, 1, 0.0001)
    _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
    # CONV LAYER 2
    _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    _mean, _var = tf.nn.moments(_conv2, [0, 1, 2])
    _conv2 = tf.nn.batch_normalization(_conv2, _mean, _var, 0, 1, 0.0001)
    _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
    # VECTORIZE
    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
    # FULLY CONNECTED LAYER 1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
    # FULLY CONNECTED LAYER 2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
    # RETURN
    out = { 'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
        'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
        'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
    }
#    print ("=== CNN READY ===")
    return out
    

# initialize CNN
def init_graph(weights,biases):    
    # FUNCTIONS
    #prediction of y_ based on x
    _pred = conv_basic(x, weights, biases, keepratio)['out']
    #cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_pred, y))
    #optimization function
    optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    #compare y_ with y
    _corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) 
    #test accuracy
    accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) 
    init = tf.initialize_all_variables()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
#    print ("=== GRAPH READY ===")
    return cost,optm,accr,sess,_corr

# train the cnn network
def train_net(trainimg,trainlabel,cnn_file_name,training_epochs,cost,optm,accr,sess):
    init = tf.initialize_all_variables()
    sess.run(init)
    print ("=== CNN TRAINING START ===")
    # SAVER
    saver = tf.train.Saver(max_to_keep=1)
    epochs_completed = 0
    index_in_epoch = 0
    
    cwd = os.getcwd()
    cnn_file = cnn_file_name + str(training_epochs-1)
    sess_ckpt_file = os.path.join(cwd,cnn_file)
    
    if (not os.path.exists(sess_ckpt_file)):
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(trainimg.shape[0]/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys, index_in_epoch, epochs_completed, trainimg, trainlabel \
                = next_batch(trainimg, trainlabel, batch_size, index_in_epoch, epochs_completed)
                # Fit training using batch data
                sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio:0.7})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})/total_batch
    
            # Display logs per epoch step
            if epoch % display_step == 0: 
                train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})
                print (" Epoch: %03d/%03d cost: %.9f, Training accuracy: %.3f" % (epoch, training_epochs, avg_cost, train_acc))
    #            test_acc = sess.run(accr, feed_dict={x: testimg[0:1000], y: testlabel[0:1000], keepratio:1.})
    #            print (" Test accuracy: %.3f" % (test_acc))
    
            # Save Net
            if epoch == training_epochs-1:
                cnn_step_file = cnn_file_name + str(epoch)
                saver.save(sess, os.path.join(cwd,cnn_step_file))
        print ("=== CNN TRAINING FINISHED ===")
    else:
        print (" loading already trained network from file %s" % sess_ckpt_file)
        saver.restore(sess, sess_ckpt_file)
        print ("=== CNN TRAINING LOADED ===")
   
    return sess,accr
    
# load next batch in training image
def next_batch(trainimg, trainlabel, batch_size, index_in_epoch, epochs_completed):
    """Return the next `batch_size` examples from this data set."""
    start = index_in_epoch
    index_in_epoch += batch_size
    num_examples = trainimg.shape[0];
    if index_in_epoch > num_examples:
      # Finished epoch
      epochs_completed += 1
      # Shuffle the data
      perm = np.arange(num_examples)
      np.random.shuffle(perm)
      trainimg = trainimg[perm]
      trainlabel = trainlabel[perm]
      # Start next epoch
      start = 0
      index_in_epoch = batch_size
#      print ("* permutate the training images [eoches completed %d] *" % (epochs_completed))
      assert batch_size <= num_examples
    end = index_in_epoch
    return trainimg[start:end], trainlabel[start:end], index_in_epoch, epochs_completed, trainimg, trainlabel

 
# test the CNN network
def test_net(testimg,testlabel,sess,accr):
    print ("=== CNN TESTING START ===")
    test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio:1})
    print (" TEST ERROR RATE: %.5f" % (1-test_acc))
    print ("=== CNN TESTING FINISHED ===")
    return test_acc

# test the CNN network for each class
def test_net_each_class(testimg,testlabel,sess,_corr):
    test_corr = sess.run(_corr, feed_dict={x: testimg, y: testlabel, keepratio:1})      
    binary_corr = sess.run(tf.cast(test_corr,tf.float32))
    
    print ("=== CNN TESTING EACH CLASS START ===")
    class_num = testlabel.argmax(axis=1)
    test_acc_list = [];
    sumnum = 0
    for i in range(10):        
        class_inds = np.where(class_num==i)[0]
        test_acc = np.mean(binary_corr[class_inds])
        print (" TEST ERROR RATE: %.5f FOR CLASS DIGIT %d [size=%d]" % (1-test_acc,i,class_inds.shape[0]))
        test_acc_list.append(test_acc)
        sumnum = sumnum + class_inds.shape[0]*test_acc
    print ("=== CNN TESTING EACH CLASS FINISHED ===")
    return np.array(test_acc_list)



