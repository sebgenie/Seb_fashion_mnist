
import tensorflow as tf
from tensorflow import keras

class modelTF :
    def __init__(self,train_images,train_labels,test_images,test_labels):
       # init here  
       self.train_images = train_images
       self.train_labels = train_labels
       self.test_images = test_images
       self.test_labels = test_labels

       # which type of model
       self.model = keras.Sequential(
            [
                keras.layers.Flatten(input_shape=(28,28)), #ransforms the format of the images from d-array (of 28 by 28 pixels), to a 1d-array of 28 * 28 = 784 pixels
                keras.layers.Dense(128,activation=tf.nn.relu), #fully connected with 128 nodes
                keras.layers.Dense(10,activation=tf.nn.softmax) 
            ]
        )
    
    def fitModel(self, tensorboardCallback, numEpochs=5):
        #build the model
        self.model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics= ['accuracy'])

        #self.model.fit(self.train_images,self.train_labels, epochs=numEpochs)
        #adding tensorboard call back
        self.model.fit(self.train_images, self.train_labels, epochs=numEpochs, validation_data=(self.test_images,self.test_labels),
          callbacks=[tensorboardCallback])

        

    def evaluateModel(self):    
        #evaluation
        test_loss, test_acc = self.model.evaluate(self.test_images,self.test_labels)
        print('Test accuary:', test_acc)

        return (test_loss, test_acc)

    def preditionOfModel(self):
        #make prediction
        predictions = self.model.predict(self.test_images)

        return predictions


