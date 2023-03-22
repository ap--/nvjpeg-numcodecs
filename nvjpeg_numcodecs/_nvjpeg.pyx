# distutils: language = c++

import cupy
from cupy.cuda.stream import Stream as Stream

from libc.stdint cimport intptr_t
from libcpp.vector cimport vector
from libcpp cimport bool


from ._common cimport cudaMalloc
from ._common cimport cudaFree
from ._common cimport cudaHostAlloc
from ._common cimport cudaFreeHost
from ._common cimport cudaError_t
from ._common cimport cudaStreamSynchronize


_status_error_msg = {
    NVJPEG_STATUS_SUCCESS: "SUCCESS",
    NVJPEG_STATUS_NOT_INITIALIZED: "NOT_INITIALIZED",
    NVJPEG_STATUS_INVALID_PARAMETER: "INVALID_PARAMETER",
    NVJPEG_STATUS_BAD_JPEG: "BAD_JPEG",
    NVJPEG_STATUS_JPEG_NOT_SUPPORTED: "JPEG_NOT_SUPPORTED",
    NVJPEG_STATUS_ALLOCATOR_FAILURE: "ALLOCATOR_FAILURE",
    NVJPEG_STATUS_EXECUTION_FAILED: "EXECUTION_FAILED",
    NVJPEG_STATUS_ARCH_MISMATCH: "ARCH_MISMATCH",
    NVJPEG_STATUS_INTERNAL_ERROR: "INTERNAL_ERROR",
    NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED: "IMPLEMENTATION_NOT_SUPPORTED",
}
_cuda_error_msg = {}


cdef int dev_malloc(void **p, size_t s):
    return <int>cudaMalloc(p, s)

cdef int dev_free(void *p):
    return <int>cudaFree(p)

cdef int host_malloc(void **p, size_t s, unsigned int f):
    return <int>cudaHostAlloc(p, s, f)

cdef int host_free(void *p):
    return <int>cudaFreeHost(p)


cdef void raise_if_nvjpeg_error(status: nvjpegStatus_t, name: str = " ") nogil except+:
    if status != NVJPEG_STATUS_SUCCESS:
        with gil:
            msg = _status_error_msg[status]
            raise RuntimeError(f"nvJPEG status error:{name}{msg!r}")


cdef void raise_if_cuda_error(cuda_error: cudaError_t, name: str = " ") nogil except+:
    if cuda_error != 0:
        # msg = _cuda_error_msg[cuda_error]
        with gil:
            msg = int(cuda_error)
            raise RuntimeError(f"nvJPEG2000 cuda error:{name}{msg!r}")


cdef int cudaStreamDefault = 0x00  # Default stream flag
cdef int cudaStreamNonBlocking = 0x01  # Stream does not synchronize with stream 0 (the NULL stream)

cdef int cudaEventDefault = 0x00  # Default event flag
cdef int cudaEventBlockingSync = 0x01  # Event uses blocking synchronization
cdef int cudaEventDisableTiming = 0x02  # Event will not record timing data
cdef int cudaEventInterprocess = 0x04  # Event is suitable for interprocess use. cudaEventDisableTiming must be set


cdef class NvJpegContext:
    cdef nvjpegHandle_t handle
    cdef bool hw_decode_available
    cdef nvjpegJpegState_t nvjpeg_state
    cdef nvjpegJpegDecoder_t nvjpeg_decoder
    cdef nvjpegJpegStream_t jpeg_stream

    def __init__(self):
        # device and host allocators
        cdef nvjpegDevAllocator_t dev_allocator
        dev_allocator.dev_malloc = &dev_malloc
        dev_allocator.dev_free = &dev_free
        cdef nvjpegPinnedAllocator_t pinned_allocator
        pinned_allocator.pinned_malloc = &host_malloc
        pinned_allocator.pinned_free = &host_free

        cdef nvjpegStatus_t status
        status = nvjpegCreateEx(
            NVJPEG_BACKEND_HARDWARE,
            &dev_allocator,
            &pinned_allocator,
            0,  # NVJPEG_FLAGS_DEFAULT,
            &self.handle
        )
        if status == NVJPEG_STATUS_ARCH_MISMATCH:
            # failed to request hardware decoder
            status = nvjpegCreateEx(
                NVJPEG_BACKEND_DEFAULT,
                &dev_allocator,
                &pinned_allocator,
                0,  # NVJPEG_FLAGS_DEFAULT,
                &self.handle,
            )
            raise_if_nvjpeg_error(status)
            self.hw_decode_available = False

        else:
            raise_if_nvjpeg_error(status)
            self.hw_decode_available = True

        status = nvjpegJpegStateCreate(
                 self.handle,
                 &self.nvjpeg_state,
        )
        raise_if_nvjpeg_error(status)

        status = nvjpegDecoderCreate(
            self.handle,
            NVJPEG_BACKEND_DEFAULT,
            &self.nvjpeg_decoder,
        )
        raise_if_nvjpeg_error(status)

        status = nvjpegJpegStreamCreate(
            self.handle,
            &self.jpeg_stream,
        )

        # CHECK_NVJPEG(nvjpegBufferDeviceCreate(params.nvjpeg_handle, NULL, &params.device_buffer));
        # CHECK_NVJPEG(nvjpegDecodeParamsCreate(params.nvjpeg_handle, &params.nvjpeg_decode_params));

    def __dealloc__(self):
        cdef nvjpegStatus_t status

        if self.jpeg_stream != NULL:
            status = nvjpegJpegStreamDestroy(self.jpeg_stream)
            raise_if_nvjpeg_error(status)

        if self.nvjpeg_decoder != NULL:
            status = nvjpegDecoderDestroy(self.nvjpeg_decoder)
            raise_if_nvjpeg_error(status)

        if self.nvjpeg_state != NULL:
            status = nvjpegJpegStateDestroy(self.nvjpeg_state)
            raise_if_nvjpeg_error(status)

        if self.handle != NULL:
            status = nvjpegDestroy(self.handle)
            raise_if_nvjpeg_error(status)


cdef class NvJpegDecodeParams:
    cdef nvjpegDecodeParams_t ptr

    def __init__(self, ctx: NvJpegContext):
        status = nvjpegDecodeParamsCreate(
            ctx.handle,
            &self.ptr,
        )
        raise_if_nvjpeg_error(status)

        # status = nvjpeg2kDecodeParamsSetRGBOutput(self.ptr, rgb_output)
        # try:
        #     raise_if_nvjpeg2k_error(status)
        # except RuntimeError:
        #     self.__dealloc__()
        #     raise

    def __dealloc__(self):
        if self.ptr != NULL:
            status = nvjpegDecodeParamsDestroy(self.ptr)
            raise_if_nvjpeg_error(status, "paramsDestroy")


def nvjpeg_decode(
    buf,
    out=None,
    *,
    ctx: NvJpegContext = None,
    decode_params: NvJpegDecodeParams = None,
    stream: Stream = None,
):
    cdef nvjpegStatus_t status
    cdef cudaStream_t cuda_stream

    cdef unsigned char* buffer
    cdef size_t length

    cdef int is_supported = -1

    # cdef int bytes_per_element = 1
    cdef nvjpegImage_t output_image
    # cdef nvjpeg2kImageInfo_t image_info

    # cdef vector[nvjpeg2kImageComponentInfo_t] image_comp_info
    # cdef vector[void *] decode_output_pixel_data
    # cdef vector[size_t] decode_output_pitch

    if buf is out:
        raise ValueError("cannot decode in-place")

    buffer = buf
    length = len(buf)

    # if ctx is None:
    #     ctx = NvJpeg2kContext()

    if stream is None:
        stream = Stream(non_blocking=True)
    cuda_stream = <cudaStream_t> <intptr_t> stream.ptr

    with nogil:
        cudaStreamSynchronize(cuda_stream)

        '''
        if ctx.hw_decode_available:
            nvjpegJpegStreamParseHeader(
                ctx.handle,
                <const unsigned char *> buffer,
                length,
                ctx.jpeg_stream
            )
            nvjpegDecodeBatchedSupportedEx(
                ctx.handle,
                ctx.jpeg_stream,
                decode_params.ptr,
                &is_supported,
            )
        '''

        '''
  std::vector<const unsigned char*> batched_bitstreams;
  std::vector<size_t> batched_bitstreams_size;
  std::vector<nvjpegImage_t>  batched_output;

  // bit-streams that batched decode cannot handle
  std::vector<const unsigned char*> otherdecode_bitstreams;
  std::vector<size_t> otherdecode_bitstreams_size;
  std::vector<nvjpegImage_t> otherdecode_output;

  if(params.hw_decode_available){
    for(int i = 0; i < params.batch_size; i++){
      // extract bitstream meta data to figure out whether a bit-stream can be decoded
      nvjpegJpegStreamParseHeader(params.nvjpeg_handle, (const unsigned char *)img_data[i].data(), img_len[i], params.jpeg_streams[0]);
      int isSupported = -1;
      nvjpegDecodeBatchedSupported(params.nvjpeg_handle, params.jpeg_streams[0], &isSupported);

      if(isSupported == 0){
        batched_bitstreams.push_back((const unsigned char *)img_data[i].data());
        batched_bitstreams_size.push_back(img_len[i]);
        batched_output.push_back(out[i]);
      } else {
        otherdecode_bitstreams.push_back((const unsigned char *)img_data[i].data());
        otherdecode_bitstreams_size.push_back(img_len[i]);
        otherdecode_output.push_back(out[i]);
      }
    }
  } else {
    for(int i = 0; i < params.batch_size; i++) {
      otherdecode_bitstreams.push_back((const unsigned char *)img_data[i].data());
      otherdecode_bitstreams_size.push_back(img_len[i]);
      otherdecode_output.push_back(out[i]);
    }
  }

    if(batched_bitstreams.size() > 0)
     {
          CHECK_NVJPEG(
               nvjpegDecodeBatchedInitialize(params.nvjpeg_handle, params.nvjpeg_state,
                                            batched_bitstreams.size(), 1, params.fmt));

         CHECK_NVJPEG(nvjpegDecodeBatched(
             params.nvjpeg_handle, params.nvjpeg_state, batched_bitstreams.data(),
             batched_bitstreams_size.data(), batched_output.data(), params.stream));
     }

    if(otherdecode_bitstreams.size() > 0)
    {
          CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(params.nvjpeg_decoupled_state, params.device_buffer));
          int buffer_index = 0;
          CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(params.nvjpeg_decode_params, params.fmt));
          for (int i = 0; i < params.batch_size; i++) {
              CHECK_NVJPEG(
                  nvjpegJpegStreamParse(params.nvjpeg_handle, otherdecode_bitstreams[i], otherdecode_bitstreams_size[i],
                  0, 0, params.jpeg_streams[buffer_index]));

              CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(params.nvjpeg_decoupled_state,
                  params.pinned_buffers[buffer_index]));

              CHECK_NVJPEG(nvjpegDecodeJpegHost(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                  params.nvjpeg_decode_params, params.jpeg_streams[buffer_index]));

              CHECK_CUDA(cudaStreamSynchronize(params.stream));

              CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                  params.jpeg_streams[buffer_index], params.stream));

              buffer_index = 1 - buffer_index; // switch pinned buffer in pipeline mode to avoid an extra sync

              CHECK_NVJPEG(nvjpegDecodeJpegDevice(params.nvjpeg_handle, params.nvjpeg_decoder, params.nvjpeg_decoupled_state,
                  &otherdecode_output[i], params.stream));

          }
    }
    '''
    '''
    if decode_params is None:
        decode_params = NvJpeg2kDecodeParams()


        status = nvjpeg2kStreamParse(
            ctx.handle,
            <unsigned char*> buffer,
            length,
            0,
            0,
            ctx.jpeg2k_stream
        )
        raise_if_nvjpeg2k_error(status)

        status = nvjpeg2kStreamGetImageInfo(ctx.jpeg2k_stream, &image_info)
        raise_if_nvjpeg2k_error(status)

        image_comp_info.resize(image_info.num_components)
        for c in range(image_info.num_components):
            status = nvjpeg2kStreamGetImageComponentInfo(
                ctx.jpeg2k_stream,
                &image_comp_info[c],
                c
            )
            raise_if_nvjpeg2k_error(status)

        decode_output_pixel_data.resize(image_info.num_components)
        output_image.pixel_data = decode_output_pixel_data.data()
        decode_output_pitch.resize(image_info.num_components)
        output_image.pitch_in_bytes = decode_output_pitch.data()
        output_image.num_components = image_info.num_components

        if 8 < image_comp_info[0].precision <= 16:
            output_image.pixel_type = NVJPEG2K_UINT16
            bytes_per_element = 2

        elif image_comp_info[0].precision == 8:
            output_image.pixel_type = NVJPEG2K_UINT8
            bytes_per_element = 1

        else:
            raise RuntimeError(f"nvJPEG2000 precision not supported: '{image_comp_info[0].precision!r}'")

        decode_output_pitch.resize(image_info.num_components)
        output_image.pitch_in_bytes = decode_output_pitch.data()
        output_image.num_components = image_info.num_components

    # shapes
    shape = (image_info.num_components, image_info.image_height, image_info.image_width)
    dtype = f"u{bytes_per_element}"

    # >>> generate output array
    if out is None:
        out = cupy.empty(shape, dtype=dtype, order="C")
    elif isinstance(out, cupy.ndarray):
        if out.shape != shape:
            raise ValueError("out has incorrect shape")
    else:
        raise NotImplementedError("todo: not implemented yet...")

    for c in range(image_info.num_components):
        output_image.pixel_data[c] = <void *> <intptr_t> out[c, :, :].data.ptr
        output_image.pitch_in_bytes[c] = image_info.image_width * bytes_per_element

    with nogil:

        # decode the image
        status = nvjpeg2kDecodeImage(
            ctx.handle,
            ctx.decode_state,
            ctx.jpeg2k_stream,
            decode_params.ptr,
            &output_image,
            cuda_stream,
        )
        raise_if_nvjpeg2k_error(status, "decodeImage")
    '''

    return out
