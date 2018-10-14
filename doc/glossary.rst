Glossary
========

**ModelInspector**
    The model inspector is the entry-point into Kepler. It orchestrates all
    activity in Kepler including but not limited to:
    
    1. saving model metadata for later inspection
    2. inspecting model parameters and training data
    3. auditing the learning process
    4. logging all training activity

**Check**
    A check, unless otherwise specified, is a test which inspects a model or
    data and looks for a specific condition. These are implemented as simple
    functions that take the model or data as inputs and return a boolean. They
    are integrated into the model at training time by adding them as decorators
    onto Keras model methods. Each check can optionally show a warning code and
    a message.

**ModelProxy**
    A model proxy is a thin wrapper around a Keras model that hijacks certain
    model methods so that it can perform operations like:

    1. running checks before training
    2. saving experiment results after training

**Experiment**
    An experiment is a single session consisting of any training that happens
    on a model. Practically, this is equivalent to monitoring everything that
    happens during a ``model.fit`` call. An experiment instance holds details
    like number of epochs an experiment ran for, metrics of the model at the
    beginning and at the end of the experiment, etc.

**Model Architecture**
    In the context of Kepler, model architecture is any representation of a
    model that contains its *qualitative* definition. This is used in multiple
    ways within Kepler - as a dict which contains layers and other attributes
    of a model, as a sparse vector that contains counts of each layer, etc.

**Architecture Matrix**
    When a model architecture is represented as a sparse vector, a collection
    of such vectors is called a architecture matrix, or archmat.

**Model Vectorizer**
    A sklearn DictVectorizer that converts a Keras model into a (sparse) vector
    containing information on layers.
