from sklearn.datasets import load_digits
from keras.models import Sequential
from keras import layers as L
from keras.utils import to_categorical
from keras.optimizers import SGD

model = Sequential()
model.add(L.Dense(32, input_shape=(64,)))
model.add(L.Activation('sigmoid'))
model.add(L.Dense(10))
model.add(L.Activation('sigmoid'))

digits = load_digits()
X = digits['data']
y = digits['target']
Y = to_categorical(y)

model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy')
model.fit(X, Y, epochs=10)
