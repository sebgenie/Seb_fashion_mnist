from __future__ import absolute_import,division, print_function,unicode_literals

import tensorflow as tf 
from tensorflow import keras
from TFmodel import modelTF 
from TFConvModel import modelConvTF
import datetime
#Helper
import numpy as np 
import matplotlib.pyplot as plt


class learnFromData:


    def __init__(self,learning_rate,log_dir,convFlag = True):

        # if you want to you convolution
        self.convFlag = convFlag
        self.log_dir = log_dir
        self.learning_rate = learning_rate

    def learningRate(self,epoch):
        """
        #will be used if we specify big epochs, specially for big dataset
        """

        if epoch == 3:
            self.learning_rate = self.learning_rate / 10
        if epoch == 20:
            self.learning_rate = self.learning_rate / 10
        if epoch == 50:
            self.learning_rate = self.learning_rate / 10

        tf.summary.scalar('learning rate',data=self.learning_rate,step=epoch)
        return self.learning_rate
        
    def learnFromDataTF(self):

        fashion_mnist = keras.datasets.fashion_mnist
        (train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        train_images.shape
        len(train_labels)

        # plt.figure()
        # plt.imshow(train_images[0])
        # plt.colorbar()
        # plt.grid(False)
        # plt.show()

        #let's scale this image to fit into a neural network 
        train_images = train_images / 255
        test_images = test_images / 255

        # #let's display 25 images
        # plt.figure(figsize=(10,10))
        # for i in range(25):
        #     plt.subplot(5,5, i+1)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.grid(False)
        #     plt.imshow(train_images[i],cmap=plt.cm.binary)
        #     plt.xlabel(class_names[train_labels[i]])
        # plt.show()

        #model 
        if self.convFlag :
            ##Add a reshape as the model expected 4 dimmensions
            train_images = train_images.reshape(-1,28,28,1)
            test_images = test_images.reshape(-1,28,28,1)
           
            model = modelConvTF(train_images,train_labels,test_images,test_labels)
        else:
            model = modelTF(train_images,train_labels,test_images,test_labels)

        #learning rate call back
        learningRateCallback = keras.callbacks.LearningRateScheduler(self.learningRate)
        #set the tensorboard callback
        self.log_dir = self.log_dir+"/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboardCallback = keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        # training is done here
        model.fitModel(learningRateCallback,tensorboardCallback,numEpochs=7)

        (test_loss, test_acc) = model.evaluateModel()
        predictions = model.preditionOfModel()

        print("prediction 0: ")
        predictions[0]
        np.argmax(predictions[0])




