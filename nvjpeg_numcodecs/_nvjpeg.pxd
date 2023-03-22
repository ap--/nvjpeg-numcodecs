
from ._common cimport cudaStream_t

cdef extern from "nvjpeg.h":

    # Maximum number of channels nvjpeg decoder supports
    DEF NVJPEG_MAX_COMPONENT = 4

    # nvJPEG status enums, returned by nvJPEG API
    ctypedef enum nvjpegStatus_t:
        NVJPEG_STATUS_SUCCESS = 0
        NVJPEG_STATUS_NOT_INITIALIZED = 1
        NVJPEG_STATUS_INVALID_PARAMETER = 2
        NVJPEG_STATUS_BAD_JPEG = 3
        NVJPEG_STATUS_JPEG_NOT_SUPPORTED = 4
        NVJPEG_STATUS_ALLOCATOR_FAILURE = 5
        NVJPEG_STATUS_EXECUTION_FAILED = 6
        NVJPEG_STATUS_ARCH_MISMATCH = 7
        NVJPEG_STATUS_INTERNAL_ERROR = 8
        NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED = 9
        NVJPEG_STATUS_INCOMPLETE_BITSTREAM = 10


    # Enums for EXIF Orientation
    cdef enum nvjpegExifOrientation:
        NVJPEG_ORIENTATION_UNKNOWN = 0
        NVJPEG_ORIENTATION_NORMAL = 1
        NVJPEG_ORIENTATION_FLIP_HORIZONTAL = 2
        NVJPEG_ORIENTATION_ROTATE_180 = 3
        NVJPEG_ORIENTATION_FLIP_VERTICAL = 4
        NVJPEG_ORIENTATION_TRANSPOSE = 5
        NVJPEG_ORIENTATION_ROTATE_90 = 6
        NVJPEG_ORIENTATION_TRANSVERSE = 7
        NVJPEG_ORIENTATION_ROTATE_270 = 8
    ctypedef nvjpegExifOrientation nvjpegExifOrientation_t


    # Enum identifies image chroma subsampling values stored inside JPEG input stream
    # In the case of NVJPEG_CSS_GRAY only 1 luminance channel is encoded in JPEG input stream
    # Otherwise both chroma planes are present
    ctypedef enum nvjpegChromaSubsampling_t:
        NVJPEG_CSS_444 = 0
        NVJPEG_CSS_422 = 1
        NVJPEG_CSS_420 = 2
        NVJPEG_CSS_440 = 3
        NVJPEG_CSS_411 = 4
        NVJPEG_CSS_410 = 5
        NVJPEG_CSS_GRAY = 6
        NVJPEG_CSS_UNKNOWN = -1

    # Parameter of this type specifies what type of output user wants for image decoding
    ctypedef enum nvjpegOutputFormat_t:
        NVJPEG_OUTPUT_UNCHANGED = 0   # return decompressed image as it is - write planar output
        NVJPEG_OUTPUT_YUV = 1         # return planar luma and chroma, assuming YCbCr colorspace
        NVJPEG_OUTPUT_Y = 2           # return luma component only, if YCbCr colorspace,
                                      #   or try to convert to grayscale,
                                      #   writes to 1-st channel of nvjpegImage_t
        NVJPEG_OUTPUT_RGB = 3         # convert to planar RGB
        NVJPEG_OUTPUT_BGR = 4         # convert to planar BGR
        NVJPEG_OUTPUT_RGBI = 5        # convert to interleaved RGB and write to 1-st channel of nvjpegImage_t
        NVJPEG_OUTPUT_BGRI = 6        # convert to interleaved BGR and write to 1-st channel of nvjpegImage_t
        NVJPEG_OUTPUT_FORMAT_MAX = 6  # maximum allowed value


    # Parameter of this type specifies what type of input user provides for encoding
    ctypedef enum nvjpegInputFormat_t:
        NVJPEG_INPUT_RGB         = 3  # Input is RGB - will be converted to YCbCr before encoding
        NVJPEG_INPUT_BGR         = 4  # Input is RGB - will be converted to YCbCr before encoding
        NVJPEG_INPUT_RGBI        = 5  # Input is interleaved RGB - will be converted to YCbCr before encoding
        NVJPEG_INPUT_BGRI        = 6  # Input is interleaved RGB - will be converted to YCbCr before encoding

    ctypedef enum nvjpegBackend_t:
        NVJPEG_BACKEND_DEFAULT = 0
        NVJPEG_BACKEND_HYBRID  = 1
        NVJPEG_BACKEND_GPU_HYBRID = 2
        NVJPEG_BACKEND_HARDWARE = 3
        NVJPEG_BACKEND_GPU_HYBRID_DEVICE = 4
        NVJPEG_BACKEND_HARDWARE_DEVICE = 5

    # Currently parseable JPEG encodings (SOF markers)
    ctypedef enum nvjpegJpegEncoding_t:
        NVJPEG_ENCODING_UNKNOWN                                 = 0x0
        NVJPEG_ENCODING_BASELINE_DCT                            = 0xc0
        NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN         = 0xc1
        NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN                 = 0xc2


    ctypedef enum nvjpegScaleFactor_t:
        NVJPEG_SCALE_NONE = 0  # decoded output is not scaled
        NVJPEG_SCALE_1_BY_2 = 1  # decoded output width and height is scaled by a factor of 1/2
        NVJPEG_SCALE_1_BY_4 = 2  # decoded output width and height is scaled by a factor of 1/4
        NVJPEG_SCALE_1_BY_8 = 3  # decoded output width and height is scaled by a factor of 1/8


    DEF NVJPEG_FLAGS_DEFAULT = 0
    DEF NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE = 1
    DEF NVJPEG_FLAGS_ENABLE_MEMORY_POOLS = 1<<1
    DEF NVJPEG_FLAGS_BITSTREAM_STRICT = 1<<2
    DEF NVJPEG_FLAGS_REDUCED_MEMORY_DECODE = 1<<3
    DEF NVJPEG_FLAGS_REDUCED_MEMORY_DECODE_ZERO_COPY = 1<<4
    DEF NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION = 1<<5


    # Output descriptor.
    # Data that is written to planes depends on output forman
    ctypedef struct nvjpegImage_t:
        unsigned char* channel[NVJPEG_MAX_COMPONENT]
        unsigned int pitch[NVJPEG_MAX_COMPONENT]


    # Prototype for device memory allocation, modelled after cudaMalloc()
    ctypedef int (*tDevMalloc)(void**, size_t)
    # Prototype for device memory release
    ctypedef int (*tDevFree)(void*)
    # Prototype for pinned memory allocation, modelled after cudaHostAlloc()
    ctypedef int (*tPinnedMalloc)(void**, size_t, unsigned int flags)
    # Prototype for device memory release
    ctypedef int (*tPinnedFree)(void*)


    ctypedef struct nvjpegDevAllocator_t:
        tDevMalloc dev_malloc
        tDevFree dev_free

    ctypedef struct nvjpegPinnedAllocator_t:
        tPinnedMalloc pinned_malloc
        tPinnedFree pinned_free


    ctypedef int(*tDevMallocV2)(void* ctx, void **ptr, size_t size, cudaStream_t stream)
    ctypedef int(*tDevFreeV2)(void* ctx, void *ptr, size_t size, cudaStream_t stream)

    ctypedef int(*tPinnedMallocV2)(void* ctx, void **ptr, size_t size, cudaStream_t stream)
    ctypedef int(*tPinnedFreeV2)(void* ctx, void *ptr, size_t size, cudaStream_t stream)


    ctypedef struct nvjpegDevAllocatorV2_t:
        tDevMallocV2 dev_malloc
        tDevFreeV2 dev_free
        void *dev_ctx

    ctypedef struct nvjpegPinnedAllocatorV2_t:
        tPinnedMallocV2 pinned_malloc
        tPinnedFreeV2 pinned_free
        void *pinned_ctx


    cdef struct nvjpegHandle
    ctypedef nvjpegHandle* nvjpegHandle_t

    cdef struct nvjpegJpegState
    ctypedef nvjpegJpegState* nvjpegJpegState_t

    cdef struct nvjpegJpegDecoder
    ctypedef nvjpegJpegDecoder* nvjpegJpegDecoder_t

    cdef nvjpegStatus_t nvjpegCreateEx(
            nvjpegBackend_t backend,
            nvjpegDevAllocator_t *dev_allocator,
            nvjpegPinnedAllocator_t *pinned_allocator,
            unsigned int flags,
            nvjpegHandle_t *handle
    )

    cdef nvjpegStatus_t nvjpegDestroy(nvjpegHandle_t handle)

    # Initalization of decoding state
    cdef nvjpegStatus_t nvjpegJpegStateCreate(
            nvjpegHandle_t handle,
            nvjpegJpegState_t *jpeg_handle
    )

    cdef nvjpegStatus_t nvjpegJpegStateDestroy(
            nvjpegJpegState_t jpeg_handle
    )

    # creates decoder state
    cdef nvjpegStatus_t nvjpegDecoderStateCreate(
            nvjpegHandle_t nvjpeg_handle,
            nvjpegJpegDecoder_t decoder_handle,
            nvjpegJpegState_t* decoder_state
    )

    # creates decoder implementation
    cdef nvjpegStatus_t nvjpegDecoderCreate(
            nvjpegHandle_t nvjpeg_handle,
            nvjpegBackend_t implementation,
            nvjpegJpegDecoder_t* decoder_handle
    )

    cdef nvjpegStatus_t nvjpegDecoderDestroy(
            nvjpegJpegDecoder_t decoder_handle
    )

    # handle that stores stream information - metadata, encoded image parameters, encoded stream parameters
    # stores everything on CPU side. This allows us parse header separately from implementation
    # and retrieve more information on the stream. Also can be used for transcoding and transfering
    # metadata to encoder
    cdef struct nvjpegJpegStream
    ctypedef nvjpegJpegStream* nvjpegJpegStream_t

    # decode parameters structure. Used to set decode-related tweaks
    cdef struct nvjpegDecodeParams
    ctypedef nvjpegDecodeParams* nvjpegDecodeParams_t

    # on return sets is_supported value to 0 if decoder is capable to handle jpeg_stream
    # with specified decode parameters
    cdef nvjpegStatus_t nvjpegDecoderJpegSupported(
            nvjpegJpegDecoder_t decoder_handle,
            nvjpegJpegStream_t jpeg_stream,
            nvjpegDecodeParams_t decode_params,
            int* is_supported
    )


    cdef nvjpegStatus_t nvjpegJpegStreamCreate(
        nvjpegHandle_t handle,
        nvjpegJpegStream_t *jpeg_stream
    )

    cdef nvjpegStatus_t nvjpegJpegStreamDestroy(
        nvjpegJpegStream_t jpeg_stream
    )


    cdef nvjpegStatus_t nvjpegDecodeParamsCreate(
        nvjpegHandle_t handle,
        nvjpegDecodeParams_t *decode_params
    )

    cdef nvjpegStatus_t nvjpegDecodeParamsDestroy(
        nvjpegDecodeParams_t decode_params
    )

    # set output pixel format - same value as in nvjpegDecode()
    cdef nvjpegStatus_t nvjpegDecodeParamsSetOutputFormat(
        nvjpegDecodeParams_t decode_params,
        nvjpegOutputFormat_t output_format
    )

    cdef nvjpegStatus_t nvjpegJpegStreamParseHeader(
        nvjpegHandle_t handle,
        const unsigned char *data,
        size_t length,
        nvjpegJpegStream_t jpeg_stream
    )

    cdef nvjpegStatus_t nvjpegDecodeBatchedSupportedEx(
        nvjpegHandle_t handle,
        nvjpegJpegStream_t jpeg_stream,
        nvjpegDecodeParams_t decode_params,
        int * is_supported
    )

    # todo: need to continue setting this up ...
