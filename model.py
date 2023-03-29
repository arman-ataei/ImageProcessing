''''
#machine learning model to recognize character of every segmented image
#Model : CNN
#Architecture: ShallowNet
#Refrence: Adrian Rosebrock, Deep Learning

'''
# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input
        # "channels last"
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels first", update
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(filters= 32,kernel_size= (5, 5), padding="same",
        input_shape=inputShape))
        model.add(Activation("relu"))
        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model
    