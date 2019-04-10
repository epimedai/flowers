import matplotlib.pyplot as plt
import numpy as np

from keras.applications.vgg19 import decode_predictions, preprocess_input, VGG19
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(1234)
batches = ImageDataGenerator(preprocessing_function=preprocess_input)
batches = batches.flow_from_directory(
    'flowers/train', target_size=(224, 224), batch_size=4)

indices = batches.class_indices
labels = [None] * 17
for key in indices:
    labels[indices[key]] = key

model = VGG19(include_top=True, input_shape=(224, 224, 3), 
              weights='imagenet')

for X, y in batches:
    preds = model.predict(X)
    decoded_preds = decode_predictions(preds, top=1)
    for i in range(len(X)):
        img = X[i].astype(np.uint8)
        label = labels[np.argmax(y[i])]
        predicted = decoded_preds[i]

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(img)
        fig.suptitle('GT: {}, Predicted: {}'.format(label, predicted))
        plt.show()
