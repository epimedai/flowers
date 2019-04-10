import matplotlib.pyplot as plt
import numpy as np

from keras.applications.vgg19 import decode_predictions, preprocess_input, VGG19
from keras.engine import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


np.random.seed(1234)

# If you run into memory errors, try reducing this
batch_size = 32

train_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input)
train_batches = train_generator.flow_from_directory('flowers/train', 
                                                    target_size=(224, 224), 
                                                    batch_size=batch_size)

val_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input)
val_batches = val_generator.flow_from_directory('flowers/val', 
                                                target_size=(224, 224), 
                                                batch_size=batch_size)

indices = train_batches.class_indices
labels = [None] * 17
for key in indices:
    labels[indices[key]] = key

pretrained = VGG19(include_top=False, input_shape=(224, 224, 3), 
                   weights='imagenet', pooling='max')

for layer in pretrained.layers:
    layer.trainable = False

inputs = pretrained.input
outputs = pretrained.output

hidden = Dense(128, activation='relu')(outputs)
preds = Dense(17, activation='softmax')(hidden)

model = Model(inputs, preds)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4), metrics=['acc'])

model.fit_generator(train_batches, 
                    epochs=100, 
                    validation_data=val_batches,
                    steps_per_epoch=len(train_batches), 
                    validation_steps=len(val_batches))
