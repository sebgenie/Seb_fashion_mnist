import tensorflow as tf
from tensorflow import keras
import numpy as np

class productionModel:
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # def __init__(self, test_image):
    #     self.test_image = test_image

    def testFromSaveModel(self,test_images):

        
        #load model from graph
        model = keras.models.load_model('saveModel/')
        #print summary
        model.summary()
        ##Add a reshape as the model expected 4 dimmensions
        test_images = test_images.reshape(-1,28,28,1)

        #print prediction
        predictions = model.predict(test_images)

         
        return self.class_names[np.argmax(predictions[0])] # as it was normalise between 0 to 1. let's get the original value for the prediction