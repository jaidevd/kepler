from keras.models import Sequential
from keras import layers as L
from keras.optimizers import SGD
from sklearn.datasets import make_classification
from kepler import ModelInspector

X, y = make_classification()

model = Sequential([
    L.Dense(16, input_shape=(20,)),
    L.Activation('sigmoid'),
    L.Dense(1),
    L.Activation('sigmoid'),
])
model.compile(loss='binary_crossentropy', optimizer=SGD())

with ModelInspector(model=model) as mi:
    mi.history = mi.fit(X, y, epochs=5).history
