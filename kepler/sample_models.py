"""Module for loading sample models."""

from keras.models import Sequential
from keras import layers as L
from keras.optimizers import SGD


def mnist_shallow(compile=True):
    """Load a shallow model meant for the sklearn MNIST dataset.

    Parameters
    ----------
        compile : bool, optional
            If true (default), the model is compiled.

    Returns
    -------
    keras model

    Example
    -------

    """
    model = Sequential([
        L.Dense(32, input_shape=(64,)),
        L.Activation('sigmoid'),
        L.Dense(10),
        L.Activation('sigmoid')
    ])
    if compile:
        model.compile(optimizer=SGD(lr=0.05), loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model
