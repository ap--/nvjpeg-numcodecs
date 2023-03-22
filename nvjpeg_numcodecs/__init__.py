"""nvjpeg numcodec for decoding directly on nvidia gpus"""
try:
    from nvjpeg_numcodecs._version import __version__
except ImportError:
    __version__ = "not-installed"

from nvjpeg_numcodecs.nvjpeg import NvJpeg

__all__ = [
    "NvJpeg",
]
