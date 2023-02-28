import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os

# Load saved model
model = keras.models.load_model('models/v.9887_my_model.h5')


def image(image_path):
    img = keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    img_array = keras.preprocessing.image.img_to_array(img)
    ##to run other you need (1, -1)or img_array = img_array.reshape(1, -1) / 255.0
    img_array = img_array.reshape(1, -1) / 255.0

    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)
    print('Predicted class:', class_id)

    # Plot the image with the predicted class label
    plt.imshow(img, cmap='gray')
    plt.title('Predicted class: {}'.format(class_id))
    plt.show()

#tset your image (image must be 28*28)
folder_path = '\image'

file_paths = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        file_paths.append(file_path)

print(file_paths)

for i in file_paths:
    image(i)
