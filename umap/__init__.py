from warnings import warn, catch_warnings, simplefilter
from .umap_ import UMAP

# Workaround: https://github.com/numba/numba/issues/3341
import numba

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("umap-learn")
except PackageNotFoundError:
    __version__ = "0.5-dev"
