from TFlearn import learnFromData
from TFtest import productionModel
from tensorflow import keras
import tensorflow as tf

production = False

def trainingModel():
    training = learnFromData(convFlag=True) #use convolution for the training
    training.learnFromDataTF()

def testModel():
    fashion_mnist = keras.datasets.fashion_mnist
    _,(test_images,test_labels) = fashion_mnist.load_data()

    
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


