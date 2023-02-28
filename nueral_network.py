import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# this part is useless you can remove it
if len(tf.config.experimental.list_physical_devices('GPU')) > 1:
    # set the GPU as the default device
    tf.config.experimental.set_visible_devices(tf.config.experimental.list_physical_devices('GPU')[1], 'GPU')
else:
    print("No GPU available")


# Load the CSV file, put the file location
data = np.loadtxt('/dataset/train.csv', delimiter=',', skiprows=1)
data_test = np.loadtxt('/dataset/test.csv', delimiter=',', skiprows=1)

# Extract the  training image dataset and labels
x_train = data[:, 1:]
y_train = data[:, 0]

# Extract the test image from the dataset
x_test = data[:, 1:]
y_test = data[:, 0]

# Reshape the image data to 28x28 pixels
# data is in 28*28*1 shape. fist variable is labels and other are image
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

# preprocess the data
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

# define the model architecture and add layers
model = keras.Sequential([
    keras.layers.Dense(128*2, activation='ReLU', input_shape=(28*28,)),
    keras.layers.Dense(128*2, activation='ReLU'),
    keras.layers.Dense(128, activation='ReLU'),
    keras.layers.Dense(10, activation='softmax')
])

# compile the model, optimizer or back propagation
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# evaluate the model on the test set. loss function
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# plot the training loss over epochs
loss = history.history['loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.legend()
plt.show()

# Save the trained model to a file
model.save('\models\*****.h5')