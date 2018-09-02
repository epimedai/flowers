import matplotlib.pyplot as plt
import numpy as np

from keras.applications.vgg19 import decode_predictions, preprocess_input, VGG19
from keras.engine import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


np.random.seed(1234)

train_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_batches = train_generator.flow_from_directory('flowers/train', target_size=(224, 224))

val_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
val_batches = val_generator.flow_from_directory('flowers/val', target_size=(224, 224))

indices = train_batches.class_indices
labels = [None] * 17
for key in indices:
    labels[indices[key]] = key

pretrained = VGG19(include_top=False, input_shape=(224, 224, 3), weights='imagenet', pooling='max')

for layer in pretrained.layers:
    layer.trainable = False

inputs = pretrained.input
outputs = pretrained.output

hidden = Dense(128, activation='relu')(outputs)
preds = Dense(17, activation='softmax')(hidden)

model = Model(inputs, preds)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['acc'])

model.fit_generator(train_batches, epochs=10, validation_data=val_batches)
"""
for X, y in generator:
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
"""