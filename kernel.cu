__constant__ float t2m;

__global__ void updatex(float* xs, float* prev_xs, float* prevprev_xs, float* fx) {    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < count_x && j < count_y) {
        int idx = i + (j * count_y);
        
        xs[idx] = (t2m * fx[idx]) + (2 * prev_xs[idx]) - prevprev_xs[idx];
    }
}
