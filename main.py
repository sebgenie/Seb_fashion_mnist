from TFlearn import learnFromData
from TFtest import productionModel
from tensorflow import keras
import tensorflow as tf

production = False
<<<<<<< HEAD

def trainingModel():
    training = learnFromData(convFlag=True) #use convolution for the training
=======
log_dir='logs'

#To stat tensorbord through command line, use:
#tensorboard --logdir logs/fit

def trainingModel():
    training = learnFromData(log_dir,convFlag=True) #use convolution for the training
>>>>>>> aeff5b3eeb0768b1ef93a48d2bf9e20084bc8479
    training.learnFromDataTF()

def testModel():
    fashion_mnist = keras.datasets.fashion_mnist
    _,(test_images,test_labels) = fashion_mnist.load_data()

<<<<<<< HEAD
    
=======
>>>>>>> aeff5b3eeb0768b1ef93a48d2bf9e20084bc8479
    test_images = test_images / 255
    predictedValue = productionModel()
    print("the image is: "+ predictedValue.testFromSaveModel(test_images))

def choice():
    if production:
        print("This is A production Version!!!")
        testModel()
    else:
        print("This is A Training !!!")
        trainingModel()

if __name__ == "__main__":
    if tf.test.is_gpu_available() :
        print("------------- Using GPU !!! -------------")
        with tf.device('/device:XLA_GPU:0'):
            choice()
    else:
        print("------------- Using CPU !!! -------------")
        choice()


