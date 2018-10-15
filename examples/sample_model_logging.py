from keras.models import Sequential
from keras import layers as L
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from kepler.main import ModelInspector


model = Sequential([
    L.Dense(32, input_shape=(64,)),
    L.Activation('sigmoid'),
    L.Dense(10),
    L.Activation('sigmoid')
])
model.compile(optimizer=SGD(lr=0.05), loss='categorical_crossentropy',
              metrics=['accuracy'])


def main():
    digits = load_digits()
    X = digits['data']
    y = digits['target']
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, stratify=y)
    ytrain, ytest = map(to_categorical, (ytrain, ytest))
    with ModelInspector(model=model) as mp:  # noqa
        mp.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=10)


if __name__ == '__main__':
    main()
