

cdef extern from "library_types.h":
    cdef enum libraryPropertyType:
        MAJOR_VERSION
        MINOR_VERSION
        PATCH_LEVEL

    ctypedef libraryPropertyType libraryPropertyType_t


cdef extern from "cuda_runtime_api.h":
    cdef enum cudaError:
        cudaSuccess = 0
        cudaErrorMissingConfiguration = 1
        cudaErrorMemoryAllocation = 2
        cudaErrorInitializationError = 3
        cudaErrorLaunchFailure = 4
        cudaErrorPriorLaunchFailure = 5
        cudaErrorLaunchTimeout = 6
        cudaErrorLaunchOutOfResources = 7
        cudaErrorInvalidDeviceFunction = 8
        cudaErrorInvalidConfiguration = 9
        cudaErrorInvalidDevice = 10
        cudaErrorInvalidValue = 11
        cudaErrorInvalidPitchValue = 12
        cudaErrorInvalidSymbol = 13
        cudaErrorMapBufferObjectFailed = 14
        cudaErrorUnmapBufferObjectFailed = 15
        cudaErrorInvalidHostPointer = 16
        cudaErrorInvalidDevicePointer = 17
        cudaErrorInvalidTexture = 18
        cudaErrorInvalidTextureBinding = 19
        cudaErrorInvalidChannelDescriptor = 20
        cudaErrorInvalidMemcpyDirection = 21
        cudaErrorAddressOfConstant = 22
        cudaErrorTextureFetchFailed = 23
        cudaErrorTextureNotBound = 24
        cudaErrorSynchronizationError = 25
        cudaErrorInvalidFilterSetting = 26
        cudaErrorInvalidNormSetting = 27
        cudaErrorMixedDeviceExecution = 28
        cudaErrorCudartUnloading = 29
        cudaErrorUnknown = 30
        cudaErrorNotYetImplemented = 31
        cudaErrorMemoryValueTooLarge = 32
        cudaErrorInvalidResourceHandle = 33
        cudaErrorNotReady = 34
        cudaErrorInsufficientDriver = 35
        cudaErrorSetOnActiveProcess = 36
        cudaErrorInvalidSurface = 37
        cudaErrorNoDevice = 38
        cudaErrorECCUncorrectable = 39
        cudaErrorSharedObjectSymbolNotFound = 40
        cudaErrorSharedObjectInitFailed = 41
        cudaErrorUnsupportedLimit = 42
        cudaErrorDuplicateVariableName = 43
        cudaErrorDuplicateTextureName = 44
        cudaErrorDuplicateSurfaceName = 45
        cudaErrorDevicesUnavailable = 46
        cudaErrorInvalidKernelImage = 47
        cudaErrorNoKernelImageForDevice = 48
        cudaErrorIncompatibleDriverContext = 49
        cudaErrorPeerAccessAlreadyEnabled = 50
        cudaErrorPeerAccessNotEnabled = 51
        cudaErrorDeviceAlreadyInUse = 54
        cudaErrorProfilerDisabled = 55
        cudaErrorProfilerNotInitialized = 56
        cudaErrorProfilerAlreadyStarted = 57
        cudaErrorProfilerAlreadyStopped = 58
        cudaErrorAssert = 59
        cudaErrorTooManyPeers = 60
        cudaErrorHostMemoryAlreadyRegistered = 61
        cudaErrorHostMemoryNotRegistered = 62
        cudaErrorOperatingSystem = 63
        cudaErrorPeerAccessUnsupported = 64
        cudaErrorLaunchMaxDepthExceeded = 65
        cudaErrorLaunchFileScopedTex = 66
        cudaErrorLaunchFileScopedSurf = 67
        cudaErrorSyncDepthExceeded = 68
        cudaErrorLaunchPendingCountExceeded = 69
        cudaErrorNotPermitted = 70
        cudaErrorNotSupported = 71
        cudaErrorHardwareStackError = 72
        cudaErrorIllegalInstruction = 73
        cudaErrorMisalignedAddress = 74
        cudaErrorInvalidAddressSpace = 75
        cudaErrorInvalidPc = 76
        cudaErrorIllegalAddress = 77
        cudaErrorInvalidPtx = 78
        cudaErrorInvalidGraphicsContext = 79
        cudaErrorNvlinkUncorrectable = 80
        cudaErrorJitCompilerNotFound = 81
        cudaErrorCooperativeLaunchTooLarge = 82
        cudaErrorStartupFailure = 0x7f
        cudaErrorApiFailureBase = 10000
    ctypedef cudaError cudaError_t

    ctypedef struct CUevent_st
    ctypedef CUevent_st* cudaEvent_t

    ctypedef struct CUstream_st
    ctypedef CUstream_st* cudaStream_t

    cdef cudaError_t cudaMalloc(void **devPtr, size_t size)
    cdef cudaError_t cudaFree(void *devPtr)
    cdef cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
    cdef cudaError_t cudaFreeHost(void *ptr)
    cdef cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)

    cdef cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
    cdef cudaError_t cudaStreamDestroy(cudaStream_t stream)
    cdef cudaError_t cudaStreamSynchronize(cudaStream_t stream) nogil

    cdef cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
    cdef cudaError_t cudaEventDestroy(cudaEvent_t event)
    cdef cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
    cdef cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
    cdef cudaError_t cudaEventSynchronize(cudaEvent_t event)


