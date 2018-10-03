import os.path as op
from traitlets import Unicode, HasTraits, TraitError
from keras.models import Sequential
from keras import layers as L
from h5py import File as H5File


class KerasModelWeights(Unicode):

    def validate(self, obj, value):
        """Overwritten from parent to ensure that the string is path to a valid keras model.
        """
        super(KerasModelWeights, self).validate(obj, value)
        if value:
            with H5File(value, 'r') as f_in:
                if 'model_config' not in f_in.attrs:
                    raise TraitError('Path {} does not contain a valid keras model.'.format(value))
        return value


class File(Unicode):

    def validate(self, obj, value):
        super(File, self).validate(obj, value)
        if value:
            if not op.isfile(value):
                raise TraitError('File {} does not exist.')
        return value


class MyTestClass(HasTraits):

    foo = KerasModelWeights()


def main():
    model = Sequential()
    model.add(L.Dense(32, input_shape=(64,)))
    model.add(L.Activation('sigmoid'))
    model.add(L.Dense(10))
    model.add(L.Activation('sigmoid'))
    model.save('/tmp/foo.h5')
    MyTestClass(foo='/tmp/foo.h5')


if __name__ == '__main__':
    main()
