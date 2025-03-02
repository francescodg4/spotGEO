# Keras script

import pickle
import numpy as np
import keras.backend as K
from data_generator import training_generator, test_generator
from keras import models
from keras import layers

print('Training length', len(training_generator) )
print('Test length', len(test_generator) )

# Design model
model = models.Sequential()
model.add(layers.Conv2D(1, kernel_size=(3,3), padding='same', input_shape=(480, 640, 1)))

model.summary()

alpha = 0.8

def asymmetric_mse(y_true, y_pred):
    d = y_true - y_pred
    w = K.abs(alpha - K.ones((480, 640, 1))*K.cast(d < 0, 'float32'))
    return K.mean(1e6*w*K.square(d))

model.compile(optimizer='adam', loss=asymmetric_mse)

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=test_generator)

model.save('/tmp/model.h5')

X = test_generator[3]
# y = model.predict(X)

# Iout = y[0, :, :, 0]

# plt.imshow(Iout)
# plt.show()
