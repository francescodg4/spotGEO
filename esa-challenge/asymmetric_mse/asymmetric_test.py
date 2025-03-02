import numpy as np
from data_generator import training_generator, test_generator
from keras.models import load_model
# from asymmetric_mse_train import asymmetric_mse
import sys
import keras.losses
import keras.backend as K
import matplotlib.pyplot as plt


print('Training length', len(training_generator) )
print('Test length', len(test_generator) )

alpha = 0.8

def asymmetric_mse(y_true, y_pred):
    d = y_true - y_pred
    w = K.abs(alpha - K.ones((480, 640, 1))*K.cast(d < 0, 'float32'))
    return K.mean(1e6*w*K.square(d))

keras.losses.asymmetric_mse = asymmetric_mse

model = load_model(sys.argv[1])

X, y_true = test_generator[3]

print(X.shape)

y_pred = model.predict(X)

Itrue = y_true[0, :, :, 0]
Ipred = y_pred[0, :, :, 0]

plt.imshow(np.hstack([Itrue, Ipred]))
plt.show()
