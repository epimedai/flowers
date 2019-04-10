import matplotlib.pyplot as plt
import numpy as np

from keras.applications.vgg19 import decode_predictions, VGG19
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(1234)
generator = ImageDataGenerator()
batches = generator.flow_from_directory(
    'flowers/train', target_size=(224, 224), batch_size=1)

indices = batches.class_indices
labels = [None] * 17
for key in indices:
    labels[indices[key]] = key

model = VGG19(include_top=True, input_shape=(224, 224, 3), 
              weights='imagenet')

for X, y in batches:
    preds = model.predict(X)
    decoded_preds = decode_predictions(preds, top=1)
    fig = plt.figure()

    img = X[0].astype(np.uint8)
    label = labels[np.argmax(y[0])]
    predicted = decoded_preds[0]

    plt.imshow(img)
    fig.suptitle('Truth: {}, Predicted: {}'.format(label, predicted))
    plt.show()

    break
