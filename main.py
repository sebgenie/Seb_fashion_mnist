
from TFlearn import learnFromData
from TFtest import productionModel
from tensorflow import keras

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

if __name__ == "__main__":
    if production:
        print("This is A production Version!!!")
        testModel()
    else:    
      trainingModel()