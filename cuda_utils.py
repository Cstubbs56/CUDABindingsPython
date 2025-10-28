import numpy as np

from cuda.bindings import driver, nvrtc

def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
    
def get_pointer(h_x, buffersize, stream):
    d_class = checkCudaErrors(driver.cuMemAlloc(buffersize))

    # (might need cuda.bindings.driver.cuMemcpy2D?)
    checkCudaErrors(driver.cuMemcpyHtoDAsync(d_class, int(h_x.ctypes.data), buffersize, stream))

    d_x = np.array([int(d_class)], dtype=np.uint64)

    return d_class, d_x

def set_constant(d_x, h_x, stream):
    checkCudaErrors(driver.cuMemcpyHtoDAsync(d_x[0], int(h_x.ctypes.data), d_x[1], stream))
    
def convert_args(args):
    new_args = np.zeros(len(args), dtype=np.uint64)
    for i in range(len(args)):
        arg = args[i]
        if type(arg) == type(driver.CUdeviceptr()):
            new_args[i] = arg
        else:
            new_args[i] = arg.ctypes.data
    return new_args

def get_func_names(ptx):
    names = []
    
    ptxstr = ptx[0].decode('utf-8')
    ptxlist = ptxstr.split("// .globl")[1:]
    
    for i in range(len(ptxlist)):
        a = ptxlist[i]
        b = a.split(".const")[0].strip()
        c = b.split(".visible")[0].strip()
        
        names.append(c)
    
    return names

def deallocate(d_class):
    checkCudaErrors(driver.cuMemFree(d_class))
                
def cleanup(module, context, stream):      
    checkCudaErrors(driver.cuStreamDestroy(stream))
        
    checkCudaErrors(driver.cuModuleUnload(module))
    checkCudaErrors(driver.cuCtxDestroy(context))
        
"""
checkCudaErrors(driver.cuLaunchKernel(
                self.update_pos,
                self.NUM_BLOCKS,  # grid x dim
                1,  # grid y dim
                1,  # grid z dim
                self.NUM_THREADS_X,  # block x dim
                self.NUM_THREADS_Y,  # block y dim
                1,  # block z dim
                0,  # dynamic shared memory
                self.pos_stream,  # stream
                self.pos_x_args.ctypes.data,  # kernel arguments
                0,  # extra (ignore)
                ))
"""
