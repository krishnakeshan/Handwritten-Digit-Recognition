#a simple approach to handwritten digit recognition using deep learning

#import data, and models
from keras.datasets import mnist
from keras import models, layers

#split data into training and testing
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

//create a Sequential model i.e. a linear stack of "layers"
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,))) #add a dense layer with a "ReLU" activation function
network.add(layers.Dense(10, activation='softmax')) #add another dense layer with softmax activation

#prep the model for training, monitoring only accuracy
network.compile(
optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

#reshape data to 60000 arrays of 28*28 (=784) values that have values [0-1]
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

#reshape test data to 10000 arrays of 784 values with values [0-1]
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

#categorically encode labels
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#fit!
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#if you'd like to evaluate your model, uncomment the next 3 lines
#(test_loss, test_accuracy) = network.evaluate(test_images, test_labels)
#print(test_loss)
#print(test_accuracy)
