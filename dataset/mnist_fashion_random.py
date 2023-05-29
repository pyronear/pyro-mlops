import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

(x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
random_indices = np.random.choice(len(x_train), size=10, replace=False)
selected_images = x_train[random_indices]

for i, image in enumerate(selected_images):
    img = Image.fromarray(image)
    img.save(f'mnist_fashion/image_{i}.jpg')
