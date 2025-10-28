import numpy as np

from cuda.bindings import driver, nvrtc

import ./cuda_utils as cuda
from ./cuda_utils import checkCudaErrors
from ./cuda_utils import get_pointer
from ./cuda_utils import convert_args
from ./cuda_utils import set_constant

h_xs, h_ys = np.meshgrid(x, y)
h_zs = np.zeros(count, dtype=np.float32)

h_init_xs = h_xs.copy()
h_init_ys = h_ys.copy()

# Set up previous positions
h_prev_xs = h_xs
h_prev_ys = h_ys
h_prev_zs = h_zs

h_prevprev_xs = h_xs
h_prevprev_ys = h_ys
h_prevprev_zs = h_zs

# Set up inital velocities
h_vel_x = np.zeros(count)
h_vel_y = np.zeros(count)
h_vel_z = np.zeros(count)

# Forces
h_Fx = np.zeros(count)
h_Fy = np.zeros(count)
h_Fz = np.zeros(count)

# Assign constants
h_t2m = np.array(((h_t**2) / particle_mass), dtype=np.float32)
h_count_x = np.array(count[0], dtype=np.uint32)
h_count_y = np.array(count[1], dtype=np.uint32)

#
#
#                            CUDA INITIALIZATION
#
#

# Store positions and force on gpu, just pass pointer from cpu
with open('.\kernel.cu', 'r') as file:
    kernel = file.read()
    #print(kernel)

# Initialize CUDA Driver API
checkCudaErrors(driver.cuInit(0))

# Retrieve handle for device 0
cuDevice = checkCudaErrors(driver.cuDeviceGet(0))

# Derive target architecture for device 0
major = checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice))
minor = checkCudaErrors(driver.cuDeviceGetAttribute(driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice))
arch_arg = bytes(f'--gpu-architecture=compute_{major}{minor}', 'ascii')

# Create program
prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(kernel), b"kernel.cu", 0, [], []))

# Compile program (fmad allows addition and multiplication to be done at same time,
# can help performance with maybe a bit different numbers, prob worth though)
opts = [b"--fmad=true", arch_arg]
checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, 2, opts))

# Get PTX from compilation
ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
ptx = b" " * ptxSize
checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))

# Create context
params = driver.CUctxCreateParams()
context = checkCudaErrors(driver.cuCtxCreate(params, 0, cuDevice))

# Load PTX as module data and retrieve function
ptx = np.char.array(ptx)
module = checkCudaErrors(driver.cuModuleLoadData(ptx.ctypes.data))

# Max threads per block of 1024 or (32, 32, 1) if 2D
# Try for 576 per block or (24, 24, 1)
if (count_x + count_y >= 48):
    if count_x >= 24:
        NUM_THREADS_X = 24
    if count_y >= 24:
        NUM_THREADS_Y = 24
else:
    if count_x < 24:
        NUM_THREADS_X = count_x
    if count_y < 24:
        NUM_THREADS_Y = count_y

# Might be more efficient ways to do this but this makes sense to me
NUM_BLOCKS = np.ceil((count_x * count_y)/(NUM_THREADS_X * NUM_THREADS_Y))

x_stream = checkCudaErrors(driver.cuStreamCreate(0))
y_stream = checkCudaErrors(driver.cuStreamCreate(0))
z_stream = checkCudaErrors(driver.cuStreamCreate(0))

# compiled names are different than the kernel names
f_names = cuda.get_func_names(ptx)

#
#
#                    GPU POINTER INITIALIZATION
#
#
# Pull constant pointers
d_t2m = checkCudaErrors(driver.cuModuleGetGlobal(module, b"t2m"))
d_count_x = checkCudaErrors(driver.cuModuleGetGlobal(module, b"count_x"))
d_count_y = checkCudaErrors(driver.cuModuleGetGlobal(module, b"count_y"))

# Pull function pointers
d_update_pos = checkCudaErrors(driver.cuModuleGetFunction(module, f_names[0].encode('utf-8')))

# Allocate memory on the GPU and put pointers in variables
# Send constants to GPU
set_constant(d_t2m, h_t2m, x_stream)
set_constant(d_count_x, h_count_x, y_stream)
set_constant(d_count_y, h_count_y, z_stream)

# All array pointers
d_x_class, d_xs = get_pointer(h_xs, buffersize, x_stream)
d_y_class, d_ys = get_pointer(h_ys, buffersize, y_stream)
d_z_class, d_zs = get_pointer(h_zs, buffersize, z_stream)

d_px_class, d_prev_xs = get_pointer(h_prev_xs, buffersize, x_stream)
d_py_class, d_prev_ys = get_pointer(h_prev_ys, buffersize, y_stream)
d_pz_class, d_prev_zs = get_pointer(h_prev_zs, buffersize, z_stream)

d_ppx_class, d_prevprev_xs = get_pointer(h_prevprev_xs, buffersize, x_stream)
d_ppy_class, d_prevprev_ys = get_pointer(h_prevprev_ys, buffersize, y_stream)
d_ppz_class, d_prevprev_zs = get_pointer(h_prevprev_zs, buffersize, z_stream)

d_Fx_class, d_Fx = get_pointer(h_Fx, buffersize, z_stream)
d_Fy_class, d_Fy = get_pointer(h_Fy, buffersize, x_stream)
d_Fz_class, d_Fz = get_pointer(h_Fz, buffersize, y_stream)

#
#
#                    ARGUMENT INITIALIZATION
#
#
# Create list of arguments for our functions (need unified forces for position updates)
pos_x_args = [d_xs, d_prev_xs, d_prevprev_xs, d_Fx]
pos_y_args = [d_ys, d_prev_ys, d_prevprev_ys, d_Fy]
pos_z_args = [d_zs, d_prev_zs, d_prevprev_zs, d_Fz]

# Convert to pointers for all the arguments
pos_x_args = convert_args(pos_x_args)
pos_y_args = convert_args(pos_y_args)
pos_z_args = convert_args(pos_z_args)

#
#
#                    POSITION CALCULATION
#
#
def calculate_position():
    # update x
    checkCudaErrors(driver.cuLaunchKernel(
        d_update_pos, 
        NUM_BLOCKS, 1, 1, 
        NUM_THREADS_X, NUM_THREADS_Y, 1, 
        0, x_stream, pos_x_args.ctypes.data, 0))
    
    # update y
    checkCudaErrors(driver.cuLaunchKernel(
        d_update_pos, 
        NUM_BLOCKS, 1, 1, 
        NUM_THREADS_X, NUM_THREADS_Y, 1, 
        0, y_stream, pos_y_args.ctypes.data, 0))
    
    # update z
    checkCudaErrors(driver.cuLaunchKernel(
        d_update_pos,
        NUM_BLOCKS, 1, 1,
        NUM_THREADS_X, NUM_THREADS_Y, 1,
        0, z_stream, pos_z_args.ctypes.data, 0))

for i in range(frames):
    for ___ in range(iter_per_frame):
        # Synchronize streams before performing calculations
        checkCudaErrors(driver.cuStreamSynchronize(x_stream))
        checkCudaErrors(driver.cuStreamSynchronize(y_stream))
        checkCudaErrors(driver.cuStreamSynchronize(z_stream))
            
        # Perform calculations
        calculate_position()
            
        # Synchronize after calculations
        checkCudaErrors(driver.cuStreamSynchronize(x_stream))
        checkCudaErrors(driver.cuStreamSynchronize(y_stream))
        checkCudaErrors(driver.cuStreamSynchronize(z_stream))
    
        # After calculations, store results in the queue
        h_xs = np.zeros(count)
        h_ys = np.zeros(count)
        h_zs = np.zeros(count)
        h_zsb = np.zeros(count)
        
        checkCudaErrors(driver.cuMemcpyDtoHAsync(h_xs.ctypes.data, d_x_class, buffersize, x_stream))
        checkCudaErrors(driver.cuMemcpyDtoHAsync(h_ys.ctypes.data, d_y_class, buffersize, y_stream))
        checkCudaErrors(driver.cuMemcpyDtoHAsync(h_zs.ctypes.data, d_z_class, buffersize, z_stream))
        
        checkCudaErrors(driver.cuStreamSynchronize(x_stream))
        checkCudaErrors(driver.cuStreamSynchronize(y_stream))
        checkCudaErrors(driver.cuStreamSynchronize(z_stream))
