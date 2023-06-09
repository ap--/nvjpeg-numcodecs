from typing import Optional
from typing import Union

import cupy as cp
import numpy as np
from numcodecs.abc import Codec

from nvjpeg_numcodecs._nvjpeg import NvJpegContext
from nvjpeg_numcodecs._nvjpeg import NvJpegDecodeParams
from nvjpeg_numcodecs._nvjpeg import Stream
from nvjpeg_numcodecs._nvjpeg import nvjpeg_decode

# waiting for: https://peps.python.org/pep-0688/
BufferLike = Union[bytes, bytearray, memoryview]


class NvJpeg(Codec):
    """NvJpeg codec for numcodecs"""

    codec_id = "nvjpeg"

    def __init__(self, blocking: bool = False) -> None:
        self.blocking = bool(blocking)
        self._ctx = NvJpegContext()
        self._decode_params = NvJpegDecodeParams()
        self._stream = Stream(non_blocking=not blocking)

    def encode(self, buf: BufferLike) -> None:
        """Encode data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Data to be encoded. Can be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : buffer-like
            Encoded data. Can be any object supporting the new-style buffer
            protocol.
        """
        raise NotImplementedError("todo")

    def decode(self, buf: BufferLike, out: Optional[BufferLike] = None) -> BufferLike:
        """Decode data in `buf`.

        Parameters
        ----------
        buf : buffer-like
            Encoded data. Can be any object supporting the new-style buffer
            protocol.
        out : buffer-like, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer
            must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : buffer-like
            Decoded data. Can be any object supporting the new-style
            buffer protocol.
        """
        return nvjpeg_decode(  # type: ignore
            buf,
            out=_flat(out),
            ctx=self._ctx,
            decode_params=self._decode_params,
            stream=self._stream,
        )


# from imagecodecs.numcodecs import _flat
def _flat(out: Optional[BufferLike]) -> Optional[BufferLike]:
    """Return numpy array as contiguous view of bytes if possible."""
    if out is None:
        return None
    elif isinstance(out, (np.ndarray, cp.ndarray)):
        return out
    else:
        raise NotImplementedError("todo")
    # view = memoryview(out)
    # if view.readonly or not view.contiguous:
    #     return None
    # return view.cast("B")
