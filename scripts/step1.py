import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.image import ImageDataGenerator


generator = ImageDataGenerator()
batches = generator.flow_from_directory('flowers/train', batch_size=4)

indices = batches.class_indices
labels = [None] * 17
for key in indices:
    labels[indices[key]] = key

for X, y in batches:
    fig, ax = plt.subplots(1, 4)

    for i in range(len(X)):
        img = X[i].astype(np.uint8)
        label = labels[np.argmax(y[i])]

        ax[i].imshow(img)
        ax[i].set_title(label)
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.show()
    break  # We only need the first batch
