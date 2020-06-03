#include "jacobi_iteration.h"

/* Write the device kernels to solve the Jacobi iterations */

/* Use compare and swap to acquire mutex */
__device__ void lock(int *mutex)
{
    while (atomicCAS(mutex, 0, 1) != 0);
    return;
}

/* Use atomic exchange operation to release mutex */
__device__ void unlock(int *mutex)
{
    atomicExch(mutex, 0);
    return;
}

__global__ void jacobi_iteration_kernel_naive(const matrix_t A, matrix_t x, matrix_t new_x, const matrix_t B, double *ssd, int *mutex)
{
    __shared__ double ssd_per_thread[THREAD_BLOCK_SIZE];
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    /* Perform Jacobi iteration */
    double sum = -A.elements[threadID * MATRIX_SIZE + threadID] * x.elements[threadID];
    for (int i = 0; i < MATRIX_SIZE; i++)
        sum += A.elements[threadID * MATRIX_SIZE + i] * x.elements[i];
    new_x.elements[threadID] = (B.elements[threadID] - sum)/A.elements[threadID * MATRIX_SIZE + threadID];

    if (threadID < MATRIX_SIZE)
        ssd_per_thread[threadIdx.x] = (new_x.elements[threadID] - x.elements[threadID]) * (new_x.elements[threadID] - x.elements[threadID]);
    else
        ssd_per_thread[threadIdx.x] = 0.0;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride = stride >> 1) {
        if(threadIdx.x < stride)
            ssd_per_thread[threadIdx.x] += ssd_per_thread[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        lock(mutex);
        *ssd += ssd_per_thread[0];
        unlock(mutex);
    }

    return;
}

__global__ void jacobi_iteration_kernel_optimized(const matrix_t A, matrix_t x, matrix_t new_x, const matrix_t B, double *ssd, int *mutex)
{
    __shared__ double ssd_per_thread[THREAD_BLOCK_SIZE];
    unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    /* Perform Jacobi iteration */
    double sum = -A.elements[threadID * MATRIX_SIZE + threadID] * x.elements[threadID];
    for (int i = 0; i < MATRIX_SIZE; i++)
        sum += A.elements[i * MATRIX_SIZE + threadID] * x.elements[i];
    new_x.elements[threadID] = (B.elements[threadID] - sum)/A.elements[threadID * MATRIX_SIZE + threadID];

    ssd_per_thread[threadIdx.x] = (new_x.elements[threadID] - x.elements[threadID]) * (new_x.elements[threadID] - x.elements[threadID]);
    __syncthreads();

	for (unsigned int stride = blockDim.x >> 1; stride > 0; stride = stride >> 1) {
		if(threadIdx.x < stride)
			ssd_per_thread[threadIdx.x] += ssd_per_thread[threadIdx.x + stride];
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		lock(mutex);
		*ssd += ssd_per_thread[0];
		unlock(mutex);
	}

    return;
}
