from importlib.metadata import version

__version__ = version(__package__)

from .sgd import SGD, sgd
