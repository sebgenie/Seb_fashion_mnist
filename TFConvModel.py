
import tensorflow as tf
from tensorflow import keras

class modelConvTF :
    def __init__(self,train_images,train_labels,test_images,test_labels):
       # init here  
       self.train_images = train_images
       self.train_labels = train_labels
       self.test_images = test_images
       self.test_labels = test_labels

       # which type of model
       self.model = keras.Sequential(
            [
                # keras.layers.Conv2D(64,(3,3), activation= tf.nn.relu, input_shape=(150,150,3)),
                # keras.layers.MaxPool2D(2,2),
                # keras.layers.Conv2D(64,(3,3), activation=tf.nn.relu),
                # keras.layers.MaxPool2D(2,2),
                # keras.layers.Conv2D(128,(3,3),activation=tf.nn.relu),
                # keras.layers.MaxPool2D(2,2),
                # keras.layers.Conv2D(128,(3,3),activation=tf.nn.relu),
                # keras.layers.MaxPool2D(2,2),

                keras.layers.Conv2D(32,(3,3), activation= tf.nn.relu, input_shape=(28,28,1)),
                
                keras.layers.BatchNormalization(axis=-1),
                keras.layers.Conv2D(32,(3,3), activation=tf.nn.relu),
                keras.layers.MaxPool2D(2,2),

                keras.layers.BatchNormalization(axis=-1),
                keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu),
                keras.layers.MaxPool2D(2,2),

                keras.layers.BatchNormalization(axis=-1),
                keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu),
                keras.layers.MaxPool2D(2,2),
                #let's flatten the output
                keras.layers.Flatten(),
                keras.layers.Dropout(0.5),

                #fully connected layer
                #keras.layers.Dense(512,activation=tf.nn.relu),
                keras.layers.Dense(128,activation=tf.nn.relu),
                keras.layers.Dense(10,activation=tf.nn.softmax)
            ]
        )
    
<<<<<<< HEAD
    def fitModel(self, numEpochs=5):
=======
    def fitModel(self, tensorboardCallback,numEpochs=5):
>>>>>>> aeff5b3eeb0768b1ef93a48d2bf9e20084bc8479
        #build the model
        self.model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics= ['accuracy'])

<<<<<<< HEAD
        self.model.fit(self.train_images,self.train_labels, epochs=numEpochs)

=======
        #self.model.fit(self.train_images,self.train_labels,epochs=numEpochs)
        #use tensorboardCallback
        self.model.fit(self.train_images,self.train_labels,epochs=numEpochs, validation_data=(self.test_images,self.test_labels),
          callbacks=[tensorboardCallback])
>>>>>>> aeff5b3eeb0768b1ef93a48d2bf9e20084bc8479
        #save model
        self.model.save('saveModel/',save_format='tf')
        

    def evaluateModel(self):    
        #evaluation
        test_loss, test_acc = self.model.evaluate(self.test_images,self.test_labels)
        print('Test accuary:', test_acc)

        return (test_loss, test_acc)

    def preditionOfModel(self):
        #make prediction
        predictions = self.model.predict(self.test_images)

        return predictions


