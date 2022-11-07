# Import all the dependencies
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras import Input, Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
# Build the model, here the encoding dimension decides by what amount the image will compress, lesser the dimension more the compression.


encoding_dim = 392
input_img = Input(shape=(784,))
# encoded representation of input
encoded = Dense(650, activation='relu')(input_img)
encoded = Dense(520, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
# decoded representation of code
decoded = Dense(520, activation='sigmoid')(encoded)
decoded = Dense(650, activation='sigmoid')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
# Model which take input image and shows decoded images
autoencoder = Model(input_img, decoded)
# Build the encoder model decoder model separately so that we can easily differentiate between input and output

# This model shows encoded images
print(input_img)
print(encoded)
print(decoded)
encoder = Model(input_img, encoded)
# Creating a decoder model
encoded_input = Input(shape=(encoding_dim,))
# last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# decoder model
decoder = Model(encoded_input, decoded)
# Compile the model with Adam optimizer and cross entropy loss function, fitment
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print(autoencoder)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))
autoencoder.fit(x_train, x_train,
                epochs=15,
                batch_size=256,
                validation_data=(x_test, x_test))
img = x_train[0, :].reshape(28, 28)  # First image in the training set.
plt.imshow(img, cmap='gray')
plt.show()  # Show the image
encoded_img = encoder.predict(x_test)
print(encoded_img[0])
decoded_img = decoder.predict(encoded_img)
print(decoded_img.shape)
img = decoded_img[0, :].reshape(28, 28)  # First image in the training set.
plt.imshow(img, cmap='gray')
plt.show()  # Show the image
