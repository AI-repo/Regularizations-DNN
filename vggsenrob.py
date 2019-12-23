## Adaptive sen-rob CRC CSV file, 4855 classes

#########################         Libraries  & packeges          #######################################
import numpy as np
import os, cv2
#import sys
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
import csv
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import re
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
#sys.path.append('/data/Amin/code')

img_size = (224, 224, 3)
batch_size = 32
epochs = 5

##################################              Curate dataset into CSV file              ##############################

data_path = '/home/amin/Datasets/CRC/1'
folders_list = [os.path.join(data_path, x) for x in sorted(os.listdir(data_path), key=natural_key)]
Unicode_list = sorted(os.listdir(data_path), key = natural_key)

def prepare_CSV (folders_list, Unicode_list):
    count = 0
    for folder_path in folders_list: 
        images_path = np.array([os.path.join(folder_path , x) for x in sorted(os.listdir(folder_path) , key= natural_key)])
        nimg_class = len(images_path)
        target = np.repeat(count, nimg_class)
        Class_Unicode = np.repeat(Unicode_list[count-1], nimg_class)
    
        if count == 0 : 
            images_add = images_path
            targets = target
            Unicodes = Class_Unicode
        else: 
            images_add = np.concatenate((images_add, images_path), 0)
            targets = np.concatenate((targets, target),0)
            Unicodes = np.concatenate((Unicodes, Class_Unicode), 0)
        count += 1
        
#    stack1 = np.vstack((Unicodes, images_add))
#    stack2 = np.vstack((stack1, targets))
    stack2 = np.vstack((images_add, targets))
    CSV_tensor = np.transpose(stack2)
    header = np.reshape(np.array([["Addresses", "Targets"]]),[1,-1]) 

    myfile = open('/home/amin/Datasets/CRC/crc_csv.csv', 'w')
    with myfile:
        writer = csv.writer(myfile)
        writer.writerows(header)
        writer.writerows(CSV_tensor)
    print("Writing complete")
    return CSV_tensor

csv_output = prepare_CSV (folders_list, Unicode_list)
##############################################################
pretrained_weight_path = '/home/amin/models/pretrained_weight'
pretrained_address_list = [ pretrained_weight_path + '/%s' %x for x in sorted(os.listdir(pretrained_weight_path),key=natural_key)]
pretrained_weights = [np.load(x) for x in pretrained_address_list]

def read_images(img_path, img_size, batch_size):
    images = np.array([ cv2.resize( cv2.imread(x, 1), (img_size[0], img_size[1]) ) for x in img_path ])
    return np.reshape( images, [ batch_size, img_size[0], img_size[1], img_size[2]])


###########################                    Fetch the test dataset                #########################33

model_path = '/home/amin/models/tuned_weights'
def get_seg_images_path(seg_subpath, Unicodes):
    count = 1 
    for paths in seg_subpath:
        #	print(natsorted(os.listdir(folder_path+mode)))
        path_Unicode = os.path.split(paths)[-1].split('_')[-1]
        try:
            target_index = Unicodes.index(path_Unicode)
        except:
            continue
        
        images_path = np.array([os.path.join(paths , x) for x in sorted(os.listdir(paths), key=natural_key)])
        images_path = np.reshape(images_path, [-1])     
        nimg_class = len(images_path)
        target = np.repeat(target_index, nimg_class)
 
        if count == 1 :
            images_address = images_path
            targets = target
        else :
            images_address = np.concatenate((images_address, images_path), 0)
            targets = np.concatenate((targets, target), 0)
        count += 1
        #print(count)
    return images_address, targets
    
#######################          VGG structure/train process            ###############################
class vgg16(object):
    def __init__(self, img_size, n_classes, batch_size, pretrained_weights):
        #n_classes=2 in case of ECG signal, positive & negative
        self.img_size = (img_size[0], img_size[1], img_size[2])
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.pretrained_weights = pretrained_weights
        self.sen_coe_initialvalue = 10
        self.rob_coe_initialvalue = 7
        self.lamda = 0.7
        self.convolution_model()
        
    
    def convolution_model(self):
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_size[2]])
        self.targets = tf.placeholder(tf.int32, [self.batch_size, 1])
        self.fdot_prev = tf.placeholder(tf.float32, [self.batch_size, self.n_classes])
        
        # conv1_1 ####################################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[0], dtype = tf.float32, shape = [3, 3, self.img_size[2], 64]),
                             trainable=True, name='conv1_1_weights')
        conv = tf.nn.conv2d(self.images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[1], dtype = tf.float32, shape = [64]), 
                             trainable=True, name='conv_1_1_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations1 = tf.nn.relu(out)

        # conv1_2 ####################################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[2], dtype = tf.float32, shape = [3, 3, 64, 64]), 
                             trainable=True, name='conv1_2_weights')
        conv = tf.nn.conv2d(self.activations1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[3], dtype = tf.float32, shape = [64]), 
                             trainable=True, name='conv1_2_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations2 = tf.nn.relu(out)

        # pool1 ######################################
        self.pool1 = tf.nn.max_pool(self.activations2, 
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1 ###################################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[4], dtype = tf.float32, shape = [3, 3, 64, 128]),
                             trainable=True, name='conv2_1_weights')
        conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[5], dtype = tf.float32, shape = [128]),
                             trainable=True, name='conv2_1_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations3 = tf.nn.relu(out)

        # conv2_2 ##################################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[6], dtype = tf.float32, shape = [3, 3, 128, 128]),
                             trainable=True, name='conv2_2_weights')
        conv = tf.nn.conv2d(self.activations3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[7], dtype = tf.float32, shape = [128]),
                             trainable=True, name='conv2_2_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations4 = tf.nn.relu(out)
            
        # pool2 #####################################
        self.pool2 = tf.nn.max_pool(self.activations4,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1 ###################################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[8], dtype = tf.float32, shape = [3, 3, 128, 256]),
                             trainable=True, name='conv3_1_weights')
        conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[9], dtype = tf.float32, shape = [256]),
                             trainable=True, name='conv3_1_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations5 = tf.nn.relu(out)

        # conv3_2 ##################################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[10], dtype = tf.float32, shape = [3, 3, 256, 256]), 
                             trainable=True, name='conv3_2_weights')
        conv = tf.nn.conv2d(self.activations5, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[11], dtype = tf.float32, shape = [256]),
                             trainable=True, name='conv3_2_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations6 = tf.nn.relu(out)
           
        # conv3_3 #################################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[12], dtype = tf.float32, shape = [3, 3, 256, 256]),
                             trainable=True, name='conv3_3_weights')
        conv = tf.nn.conv2d(self.activations6, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[13], dtype = tf.float32, shape = [256]),
                             trainable=True, name='conv3_3_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations7 = tf.nn.relu(out)

        # pool3 ###################################
        self.pool3 = tf.nn.max_pool(self.activations7,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1 #################################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[14], dtype = tf.float32, shape = [3, 3, 256, 512]),
                             trainable=True, name='conv4_1_weights')
        conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[15], dtype = tf.float32, shape = [512]),
                             trainable=True, name='conv4_1_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations8 = tf.nn.relu(out)

        # conv4_2 ################################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[16], dtype = tf.float32, shape = [3, 3, 512, 512]),
                             trainable=True, name='conv4_2_weights')
        conv = tf.nn.conv2d(self.activations8, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[17], dtype = tf.float32, shape = [512]),
                             trainable=True, name='conv4_2_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations9 = tf.nn.relu(out)

        # conv4_3 ################################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[18], dtype = tf.float32, shape = [3, 3, 512, 512]),
                             trainable=True, name='conv4_3_weights')
        conv = tf.nn.conv2d(self.activations9, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[19], dtype = tf.float32, shape = [512]),
                             trainable=True, name='conv4_3_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations10 = tf.nn.relu(out)

        # pool4 #################################
        self.pool4 = tf.nn.max_pool(self.activations10,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',#import vgg_main as mainfile
                                    name='pool4')

        # conv5_1 ###############################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[20], dtype = tf.float32, shape = [3, 3, 512, 512]),
                             trainable=True, name='conv5_1_weights')
        conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[21], dtype = tf.float32, shape = [512]),
                             trainable=True, name='conv5_1_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations11 = tf.nn.relu(out)

        # conv5_2 ###############################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[22], dtype = tf.float32, shape = [3, 3, 512, 512]), 
                             trainable=True, name='conv5_2_weights')
        conv = tf.nn.conv2d(self.activations11, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[23], dtype = tf.float32, shape = [512]),
                             trainable=True, name='conv5_2_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations12 = tf.nn.relu(out)

        # conv5_3 ###############################
        kernel = tf.Variable(tf.constant(self.pretrained_weights[24], dtype = tf.float32, shape = [3, 3, 512, 512]),
                             trainable=True, name='conv5_3_weights')
        conv = tf.nn.conv2d(self.activations12, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(self.pretrained_weights[25], dtype = tf.float32, shape = [512]), 
                             trainable=True, name='conv5_3_biases')
        out = tf.nn.bias_add(conv, biases)
        self.activations13 = tf.nn.relu(out)

        # pool5 #################################
        self.pool5 = tf.nn.max_pool(self.activations13,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')
                               
        # fc1 ##################################
        shape = int(np.prod(self.pool5.get_shape()[1:]))
        fc1w = tf.Variable(self.pretrained_weights[26], trainable=True, name='fc1_weights')
        fc1b = tf.Variable(self.pretrained_weights[27], trainable=True, name='fc1_biases')
        pool5_flat = tf.reshape(self.pool5, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        self.activations14 = tf.nn.relu(fc1l)
        activations14_dropout = tf.nn.dropout(self.activations14, 0.7)
        
        # fc2 ##################################
        fc2w = tf.Variable(self.pretrained_weights[28], trainable=True, name='fc2_weights')
        fc2b = tf.Variable(self.pretrained_weights[29], trainable=True, name='fc2_biases')
        fc2l = tf.nn.bias_add(tf.matmul(activations14_dropout, fc2w), fc2b)
        self.activations15 = tf.nn.relu(fc2l)
        activations15_dropout = tf.nn.dropout(self.activations15, 0.7)
        
        # fc3 ##################################
        #fc3w = tf.Variable(self.pretrained_weights[30], trainable=True, name='fc3_weights')
        #fc3b = tf.Variable(self.pretrained_weights[31], trainable=True, name='fc3_biases')
#        fc3w = tf.Variable(tf.random_normal([4096, self.n_classes], stddev = 0.001, dtype=tf.float32), name='fc3_weights')        
#        fc3b = tf.Variable(tf.zeros([self.n_classes]), name='fc3_biases') 
        fc3w = tf.Variable(tf.truncated_normal([4096, self.n_classes], stddev = 0.1), name='fc3_weights')        
        fc3b = tf.Variable(tf.constant(0.1, shape=[self.n_classes]), name='fc3_biases') 
        self.fc3l= tf.nn.bias_add(tf.matmul(activations15_dropout, fc3w), fc3b)
        # No activation function (ReLU-softplus) is used for last layer
        self.prob = tf.nn.softmax(self.fc3l)
        self.onehot_labels = tf.reshape(tf.one_hot(self.targets, self.n_classes), tf.stack([self.batch_size, self.n_classes]))

        
        ###################         Sensitivity and robustness coefficients         ################################
        self.sen_coefficient = tf.Variable(tf.constant(self.sen_coe_initialvalue, dtype = tf.float32, shape = [1]),
                             trainable=True, name='sen_coe')
        self.rob_coefficient = tf.Variable(tf.constant(self.rob_coe_initialvalue, dtype = tf.float32, shape = [1]),
                             trainable=True, name='rob_coe')
               
        
        ##################           Loss & gradients of the activations (fdot)    #################################          

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc3l, labels=self.onehot_labels)
        self.loss = tf.reduce_mean(self.cross_entropy)  
  
         #        self.sensitivityterm = tf.tanh(0.00001 / (tf.reduce_mean(self.fdot_prev)+ 0.00001))
        
#        with tf.name_scope("senterm"):
        self.sensitivityterm = tf.clip_by_value(self.loss, 0, 1) /(self.sen_coefficient * tf.reduce_mean(tf.abs(self.fdot_prev))+ 1)
#            tf.summary.scalar("senterm", self.sensitivityterm)
            
#        with tf.name_scope("robterm"):
        self.robustnessterm =  self.rob_coefficient * tf.reduce_mean(self.fdot_prev)
#            tf.summary.scalar("robterm", self.robustnessterm)
        
#        with tf.name_scope("loss"):
        self.loss = (self.lamda*self.loss) + (1-self.lamda)*(tf.clip_by_value(self.robustnessterm,-0.5,5) + tf.clip_by_value(self.sensitivityterm, 0,5))
#            tf.summary.scalar("loss", self.loss)
        
#        with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax( self.prob,1), tf.argmax(self.onehot_labels,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#            tf.summary.scalar("accuracy", self.accuracy)
        
        self.activs = [self.activations1, self.activations2, self.activations3, self.activations4, self.activations5, self.activations6,
                       self.activations7, self.activations8, self.activations9, self.activations10, self.activations11, self.activations12, 
                       self.activations13, self.activations14, activations14_dropout,  self.activations15, activations15_dropout, self.fc3l]
        
        self.grad_activ = tf.gradients(self.loss, self.activs )
        self.fdot_cur = self.grad_activ[17]
        
        
 #########################                  Training                ##############################    
    def train(self):
        f = pd.read_csv('/home/amin/Datasets/CRC/crc_csv.csv', delimiter=',', encoding = 'utf-8' )
        input_data = f['Addresses'].values
        targets = f['Targets'].values
        tunedweights_path = '/home/amin/models/tuned_weights'   
        ckpt = tf.train.get_checkpoint_state(tunedweights_path)

#        optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
#        optimizer = tf.train.MomentumOptimizer(learning_rate = 0.001, momentum = 0.9).minimize(self.loss)
        
        tvars = tf.trainable_variables()
        optimizer1 = tf.train.AdamOptimizer(learning_rate = 0.0001)        
        tvars_opt1 = [var for var in tvars if not ('sen_coe' in var.name or 'rob_coe' in var.name)]
        grad_weight1 = optimizer1.compute_gradients(self.loss, tvars_opt1)
        train_step1 = optimizer1.apply_gradients(grad_weight1)
        
        optimizer2 = tf.train.AdamOptimizer(learning_rate = 0.1)      
        tvars_opt2 = [var for var in tvars if 'sen_coe' in var.name or 'rob_coe' in var.name]
        grad_weight2 = optimizer2.compute_gradients(self.loss, tvars_opt2)
        capped_grad_weight2 =  [(None if grad is None else tf.clip_by_value(grad, -1.,1.), var) for grad, var in grad_weight2]
        #tf.clip_by_norm(t,clip_norm, axes=None, name=None)
        train_step2 = optimizer2.apply_gradients(capped_grad_weight2)
                
        saver = tf.train.Saver(max_to_keep=1)
        init = tf.global_variables_initializer()    
        
        
        with tf.Session() as sess:
#            writer_train = tf.summary.FileWriter('./logs/plot_train', sess.graph)
#            merged = tf.summary.merge_all()
            sess.run(init)    
            
            # read pretrained model if there is 
            if ckpt:
                print("Reading model parameters from %s" % ckpt.model_checkpoint_path)	
                saver.restore(sess, ckpt.model_checkpoint_path)        
                epoch = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                start = int(epoch) + 1
            else : start = 0    
              
            grad_fdot = 0.3*np.ones([self.batch_size, self.n_classes])



            for epoch in range(start, epochs):
                
                #Shuffle data
                indices = np.arange(len(targets)) 
                np.random.shuffle(indices)
                input_data = input_data[indices]
                targets = targets[indices]
                
                #Perform cross-validation after shuffling
                n_splits = 3
                kf = StratifiedKFold(n_splits , shuffle=True)

                
                fold = 1
                for index, (train_indices, valid_indices) in enumerate(kf.split(input_data, targets)):
                                       
                    input_train, input_valid = input_data[train_indices], input_data[valid_indices]
                    target_train, target_valid = targets[train_indices], targets[valid_indices]                   
                    
                    ################################                    Training fold                   ################################
                    total_loss=0
                    total_accuracy=0
                    batches_train = 0
                    total_batch_train = len(input_train)/batch_size
                    for start, end in zip(range(0, len(input_train), batch_size), range(batch_size, len(input_train), batch_size)):                
                        feed_dic1 = {self.images : read_images(input_train[start:end], img_size, batch_size),
                                    self.targets : np.reshape(target_train[start:end], [batch_size, 1]),
                                    self.fdot_prev: grad_fdot}   
                        
    #                    _, batch_loss = sess.run([optimizer, loss], feed_dict = feed_dic ) 
                        _ , _, batch_loss, batch_accur, fdot_current, senterm, robterm = sess.run([train_step1 ,train_step2, self.loss , self.accuracy,
                                                                                        self.fdot_cur , self.sensitivityterm , self.robustnessterm],
                                                                                        feed_dict=feed_dic1)
                        total_loss  += batch_loss
                        total_accuracy += batch_accur
                        batches_train +=1
                        print("ep %d,    fold %d,  batch:%d|%d  ,  loss : %f,   train_accuracy : %f " 
                          %(epoch+1, fold, batches_train, total_batch_train, batch_loss, batch_accur))     
                        
                    # The following feed_dic is for updating the values with self.fdot_prev: fdot_current
                    # The previous value came from grad_fdot = np.ones([self.batch_size, self.n_classes]) now the values are updated
                    feed_dic2 = {self.images : read_images(input_train[start:end], img_size, batch_size),
                                self.targets : np.reshape(target_train[start:end], [batch_size, 1]),
                                self.fdot_prev: fdot_current }
                    
                    weights_biases, regularizers, activations, grad_activ_value , lastbatch_acc = sess.run([tvars_opt1, tvars_opt2, self.activs, self.grad_activ, 
                                                                                                            self.accuracy],feed_dict = feed_dic2 ) 
                                                                                                                                     
                    grad_fdot0 = np.array(grad_activ_value[17])
                    grad_fdot_sum = grad_fdot0.sum(axis = 0)
                    grad_fdot = grad_fdot0/np.transpose(grad_fdot_sum[:, np.newaxis])
                    
                    avg_loss = total_loss/np.round(len(input_train)/batch_size)
                    train_accuracy = total_accuracy/np.round(len(input_train)/batch_size)
                    
#                    if fold == n_splits:
#                        summary1 = sess.run(merged, feed_dict = feed_dic2)
#                        writer_train.add_summary(summary1,epoch)
#                        writer_train.flush()
                        
                    senncoef = regularizers[0]
                    robbcoef = regularizers[1]                   
                    print(" senterm : %f ,    robterm: %f,     sencoef:%f      , robcoef: %f" %(senterm, robterm, senncoef, robbcoef))
                    
#                    print("ep %d,    fold %d,     loss : %f,   train_accuracy : %f " 
#                          %(epoch+1, fold, avg_loss, train_accuracy))           
                    #save pretrained_weights            
                    saver.save(sess, os.path.join(tunedweights_path, 'model'), global_step=epoch)
                
                    
                    ###############################                    Valid fold                         ############################
                    valid_accuracy = 0.0 
                    total_valid_acc=0
                    batches_valid =0
                    total_batch_valid = len(input_valid)/batch_size

                     #last_tunedweights_path = '/home/amin/vggnet/tuned_weights/model-3'saver.restore(sess, last_tunedweights_path)
                    for start, end in zip(range(0, len(input_valid), batch_size), range(batch_size, len(input_valid), batch_size)):                
                        feed_dic_valid = {self.images : read_images(input_valid[start:end], img_size, batch_size),
                                    self.targets : np.reshape(target_valid[start:end], [batch_size, 1])} 
                                           
                        pred_ans = sess.run(self.prob , feed_dict= feed_dic_valid)
                        target_valid_b = np.array(target_valid[start:end])
                        
                        for i in range(len(target_valid_b)):  
                            
                            if (np.argmax(np.array(pred_ans[i,]),0) == int(target_valid_b[i])):    
                                valid_accuracy += 1
                        valid_accuracy /= tf.round(float(len(target_valid_b)))
                        total_valid_acc += valid_accuracy
                        batches_valid +=1
                        print(" ep %d,    fold %d,batches :%d|%d   , valid_accuracy : %f " %( epoch+1, fold, batches_valid,total_batch_valid, valid_accuracy))  
                    
                    Avg_valid_acc = total_valid_acc/np.round(len(input_valid)/batch_size)

                    if fold == 1 :
                        Ave_train_acc = train_accuracy
                        Ave_valid_acc = Avg_valid_acc
                        Ave_loss = avg_loss
                    else :
                        Ave_train_acc += train_accuracy
                        Ave_valid_acc += Avg_valid_acc
                        Ave_loss += avg_loss
            
                    fold +=1
                    
                Ave_train_acc/= n_splits
                Ave_valid_acc/= n_splits
                Ave_loss /= n_splits
                print("epoch %d, train: %f    , valid:%f:,   loss:%f     "  %( epoch+1,Ave_train_acc, Ave_valid_acc, Ave_loss))
            
            np.save('datasaved.npy', grad_fdot)
            
        return  activations, grad_activ_value, grad_fdot,regularizers,weights_biases
#    
    

#########################                  Testing                ##############################              


def test(last_tunedweights_path):
   
   #mode = '/test/'
   # input_data, target_label = get_images_path(folders_path, mode)
   # n_labels = max(target_label) + 1
   data_path = '/data/Amin/dataset/datamain'
   folders_path = [os.path.join(data_path, x) for x in sorted(os.listdir(data_path), key=natural_key)]
   Unicodes = [ os.path.split(folder)[-1].split('_')[-1] for folder in folders_path]
   #class_indices = [Unicodes.index(x) for x in Unicodes]  
    
   seg_path = '/data/Amin/seg_images'
   seg_subpath = [os.path.join(seg_path, x) for x in sorted(os.listdir(seg_path), key=natural_key)]
   images_addresses, targets = get_seg_images_path(seg_subpath, Unicodes)
#    target_label = np.reshape( np.array([288,308,1556,2010,2020,2068,2302,3406,3645,4212,308,732,1421,1556,1628,1631,2010,2068,2324,2432,
#                                         2817,2904,3058,3116,3321,3399,3406,3732,3935,288,308,1421,1576,2020,2068,2302,2339,2817,3406,                                       
#                                         4164,308,912,1208,1421,1556,1631,1808,2010,2068,2286]),[-1])
   #ckpt = tf.train.get_checkpoint_state(model_save_path)
   #n_classes = 4219
   n_labels = max(targets) + 1
   
   accuracy = 0.0 
   batch_size = 1
   argmax_prediction = np.zeros(len(images_addresses))
   targets = np.zeros(len(images_addresses)) 
   vgg = vgg16(img_size, n_labels, batch_size, pretrained_weights)
    
   with tf.Session() as sess:
   
       saver = tf.train.Saver()
       saver.restore(sess, last_tunedweights_path)
       #for start, end in zip(range(0, len(input_data), batch_size), \
       #               range(batch_size, len(input_data), batch_size)):
       for i in range(len(images_addresses)):
           test_image = read_images([images_addresses[i]], img_size, batch_size)
           test_targetvalue = np.reshape(targets[i],[1,1])
           pred_ans = sess.run(vgg.prob , feed_dict= {vgg.images : test_image, vgg.target : test_targetvalue})
           row = '{} \t {} \t {} \n'.format(test_image, test_targetvalue, pred_ans)
           with open('result.txt','w') as f:
               f.write(row)
            
            #onehot = sess.run(onehot_label, feed_dict= {x : xf, target : yf})
            #print np.argmax(ans, 1)
            #print np.argmax(onehot, 1)
           if (np.argmax(pred_ans,1) == int(test_targetvalue)):    
               accuracy += 1./float(len(images_addresses))
               print("accuracy: ", '%f' %accuracy)

           pre = np.argmax(pred_ans, 1)
           argmax_prediction[i] = pre
           b = int(test_targetvalue)    
           targets[i] = b 
          
       a=precision_recall_fscore_support(targets, argmax_prediction, average='macro')
       print("precision= ", '%04f' % (a[0]))
       print("recall= ", '%04f' % (a[1]))
       print("F meature= ", '%04f' % (a[2]))
       print("accuracy: ", '%f' %accuracy)

########################             Running train or test         ##############################

# Running train process
                            
f = pd.read_csv('/home/amin/Datasets/CRC/crc_csv.csv', delimiter=',', encoding = 'utf-8' )
input_data = f['Addresses'].values
targets = f['Targets'].values
n_classes = max(targets)+1
vgg = vgg16(img_size, n_classes, batch_size, pretrained_weights)                      
activations, grad_activ_value, grad_fdot ,regularizers,weights_biases=vgg.train()

# Running test process

#last_tunedweights_path = '/home/amin/vggnet/tuned_weights/model-3'
#test(last_tunedweights_path)


#################################################################################################


















