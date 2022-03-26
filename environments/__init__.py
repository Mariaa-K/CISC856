import platform
if platform.system() != "Windows":
    from .dmc import *
from .procgen import *
