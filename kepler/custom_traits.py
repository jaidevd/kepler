import os.path as op
from traitlets import Unicode, TraitError, List
from keras.models import Model
from h5py import File as H5File


class File(Unicode):
    """Trait representing a file."""

    def validate(self, obj, value):
        super(File, self).validate(obj, value)
        if value:
            if not op.isfile(value):
                raise TraitError('File {} does not exist.')
        return value


class KerasModelWeights(File):
    """A file containing Keras model weights."""

    def validate(self, obj, value):
        """Overwritten from parent to ensure that the string is path to a
        valid keras model.
        """
        super(KerasModelWeights, self).validate(obj, value)
        if value:
            with H5File(value, 'r') as f_in:
                if 'model_config' not in f_in.attrs:
                    raise TraitError(
                        '{} does not contain a valid keras model.'.format(
                            value))
        return value


class KerasModelMethods(List):
    """List trait containing keras model method names."""

    def validate(self, obj, values):
        super(KerasModelMethods, self).validate(obj, values)
        for method_name in values:
            func = getattr(Model, method_name, False)
            if callable(func):
                continue
            else:
                raise TraitError(method_name + ' is not a keras Model method.')
        return values
