from importlib.metadata import version

__version__ = version(__package__)

from .adam import Adam, adam
from .adamw import AdamW, adamw
from .sgd import SGD, sgd
