/* Host code for the Jacobi method of solving a system of linear equations
 * by iteration.

 * Build as follws: make clean && make

 * Author: Naga Kandasamy
 * Date: May 21, 2020

 * Modified by: Daniel Drzewicki and Brian Tu
 * Date Modified: June 3, 2020
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */
// #define DEBUG

/* Define ping-pong buffer constants*/
#define PING 0
#define PONG 1

/* Variables to measure execution time */
struct timeval start, stop;
float naive_time, opt_time;

int main(int argc, char **argv)
{
	if (argc > 1) {
		printf("This program accepts no arguments\n");
		exit(EXIT_FAILURE);
	}

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */
	matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel */
    matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel */

	/* Initialize the random number generator */
	srand(time(NULL));

	/* Generate diagonally dominant matrix */
    printf("\nGenerating %d x %d system\n", MATRIX_SIZE, MATRIX_SIZE);
	A = create_diagonally_dominant_matrix(MATRIX_SIZE, MATRIX_SIZE);
	if (A.elements == NULL) {
        printf("Error creating matrix\n");
        exit(EXIT_FAILURE);
	}

    /* Create the other vectors */
    B = allocate_matrix_on_host(MATRIX_SIZE, 1, 1);

	reference_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
	gpu_naive_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
    gpu_opt_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution on CPU */
	printf("\nPerforming Jacobi iteration on the CPU\n");
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Gold Execution time = %fs\n", (float) (stop.tv_sec - start.tv_sec\
                    + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, reference_x, B); /* Display statistics */

	/* Compute Jacobi solution on device. Solutions are returned
       in gpu_naive_solution_x and gpu_opt_solution_x. */
    printf("\nPerforming Jacobi iteration on device\n");
	compute_on_device(A, gpu_naive_solution_x, gpu_opt_solution_x, B);
    fprintf(stderr, "Naive Execution time = %fs\n", naive_time);
    display_jacobi_solution(A, gpu_naive_solution_x, B); /* Display statistics */
    fprintf(stderr, "Optimized Execution time = %fs\n", opt_time);
    display_jacobi_solution(A, gpu_opt_solution_x, B);

    free(A.elements);
	free(B.elements);
	free(reference_x.elements);
	free(gpu_naive_solution_x.elements);
    free(gpu_opt_solution_x.elements);

    exit(EXIT_SUCCESS);
}


/* Complete this function to perform Jacobi calculation on device */
void compute_on_device(const matrix_t A, matrix_t gpu_naive_sol_x,
                       matrix_t gpu_opt_sol_x, const matrix_t B)
{
    int ping_pong = 1;
    unsigned int done = 0;
    double mse = 0;
    unsigned int num_iter = 0;
    double ssd;

    gettimeofday(&start, NULL);
    /* Allocate Matrices */
    matrix_t d_A = allocate_matrix_on_device(A);
    matrix_t d_B = allocate_matrix_on_device(B);
    matrix_t d_gpu_naive_sol_x = allocate_matrix_on_device(gpu_naive_sol_x);
    matrix_t d_new_x = allocate_matrix_on_device(gpu_naive_sol_x);

    /* Allocate space for result on GPU and initialize */
    copy_matrix_to_device(d_A, A);
    copy_matrix_to_device(d_B, B);
    copy_matrix_to_device(d_gpu_naive_sol_x, B);

    /* Allocate space for the ssd on GPU and initialize it */
	double *d_ssd;
    cudaMalloc((void**)&d_ssd, sizeof(double));
    cudaMemset(d_ssd, 0.0f, sizeof(double));

    /* Allocate space for the lock on GPU and initialize it */
	int *d_mutex;
    cudaMalloc((void **)&d_mutex, sizeof(int));
    cudaMemset(d_mutex, 0, sizeof(int));

    /* Begin naive device calculation */
    dim3 threadBlock(THREAD_BLOCK_SIZE, 1, 1);
    dim3 grid(MATRIX_SIZE/THREAD_BLOCK_SIZE, 1);
    while(!done){
        ssd = 0.0f; cudaMemcpy(d_ssd, &ssd, sizeof(double), cudaMemcpyHostToDevice);
        if (ping_pong == 1)
            jacobi_iteration_kernel_naive<<<grid, threadBlock>>>(d_A, d_gpu_naive_sol_x, d_new_x, d_B, d_ssd, d_mutex);
        else
            jacobi_iteration_kernel_naive<<<grid, threadBlock>>>(d_A, d_new_x, d_gpu_naive_sol_x, d_B, d_ssd, d_mutex);
        cudaDeviceSynchronize();
        num_iter++;
        cudaMemcpy(&ssd, d_ssd, sizeof(double), cudaMemcpyDeviceToHost);
        mse = sqrt(ssd); /* Mean squared error. */
        #ifdef DEBUG
            printf("Iteration: %d. MSE = %f\n", num_iter, mse);
        #endif
        if (mse <= THRESHOLD)
            done = 1;
        ping_pong = !ping_pong;
    }
    printf("Naive Convergence achieved after %d iterations \n", num_iter);
    copy_matrix_from_device(gpu_naive_sol_x, d_gpu_naive_sol_x);

    gettimeofday(&stop, NULL);
    naive_time = (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ping_pong = 1;
    done = 0;
    mse = 0;
    num_iter = 0;

    gettimeofday(&start, NULL);
    /* Allocate Matrices */
    d_A = allocate_matrix_on_device(A);
    matrix_t A_transpose = transpose_matrix(A);
    d_B = allocate_matrix_on_device(B);
    matrix_t d_gpu_opt_sol_x = allocate_matrix_on_device(gpu_opt_sol_x);
    d_new_x = allocate_matrix_on_device(gpu_opt_sol_x);

    /* Allocate space for result on GPU and initialize */
    copy_matrix_to_device(d_A, A_transpose);
    copy_matrix_to_device(d_B, B);
    copy_matrix_to_device(d_gpu_opt_sol_x, B);

    /* Allocate space for ssd on GPU and initialize */
    cudaMalloc((void**)&d_ssd, sizeof(double));
    cudaMemset(d_ssd, 0.0f, sizeof(double));

    /* Allocate space for the lock on GPU and initialize it */
    cudaMalloc((void **)&d_mutex, sizeof(int));
    cudaMemset(d_mutex, 0, sizeof(int));

    while(!done){
        ssd = 0.0; cudaMemcpy(d_ssd, &ssd, sizeof(double), cudaMemcpyHostToDevice);
        if (ping_pong == 1)
            jacobi_iteration_kernel_optimized<<<grid, threadBlock>>>(d_A, d_gpu_opt_sol_x, d_new_x, d_B, d_ssd, d_mutex);
        else
            jacobi_iteration_kernel_optimized<<<grid, threadBlock>>>(d_A, d_new_x, d_gpu_opt_sol_x, d_B, d_ssd, d_mutex);
        cudaDeviceSynchronize();
        num_iter++;
        cudaMemcpy(&ssd, d_ssd, sizeof(double), cudaMemcpyDeviceToHost);
        mse = sqrt(ssd); /* Mean squared error. */
        #ifdef DEBUG
            printf("Iteration: %d. MSE = %f\n", num_iter, mse);
        #endif
        if (mse <= THRESHOLD)
            done = 1;
        ping_pong = !ping_pong;
    }
    printf("Opt Convergence achieved after %d iterations \n", num_iter);
    copy_matrix_from_device(gpu_opt_sol_x, d_gpu_opt_sol_x);

    gettimeofday(&stop, NULL);
    opt_time = (float) (stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_new_x.elements);
    cudaFree(d_gpu_naive_sol_x.elements);
    cudaFree(d_gpu_opt_sol_x.elements);
    return;
}

/* Allocate matrix on the device of same size as M */
matrix_t allocate_matrix_on_device(const matrix_t M)
{
    matrix_t Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void **)&Mdevice.elements, size);
    return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix_on_host(int num_rows, int num_columns, int init)
{
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;

	M.elements = (float *)malloc(size * sizeof(float));
	for (unsigned int i = 0; i < size; i++) {
		if (init == 0)
            M.elements[i] = 0;
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}

    return M;
}

/* Copy matrix to device */
void copy_matrix_to_device(matrix_t Mdevice, const matrix_t Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host */
void copy_matrix_from_device(matrix_t Mhost, const matrix_t Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Prints the matrix out to screen */
void print_matrix(const matrix_t M)
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++) {
			printf("%f ", M.elements[i * M.num_rows + j]);
        }

        printf("\n");
	}

    printf("\n");
    return;
}

/* Returns a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check for errors in kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    return;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(unsigned int num_rows, unsigned int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows;
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));
    if (M.elements == NULL)
        return M;

	/* Create a matrix with random numbers between [-.5 and .5] */
    unsigned int i, j;
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);

	/* Make diagonal entries large with respect to the entries on each row. */
	for (i = 0; i < num_rows; i++) {
		float row_sum = 0.0;
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}

        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    return M;
}

/* Returns transposed matrix */
matrix_t transpose_matrix(matrix_t M)
{
    matrix_t M_transpose;
	M_transpose.num_columns = MATRIX_SIZE;
	M_transpose.num_rows = MATRIX_SIZE;
	unsigned int size = M_transpose.num_rows * M_transpose.num_columns;
	M_transpose.elements = (float *)malloc(size * sizeof(float));
    for (int i=0; i < MATRIX_SIZE; i++) {
        for(int j=0; j < MATRIX_SIZE; j++) {
            M_transpose.elements[i*MATRIX_SIZE + j] = M.elements[j*MATRIX_SIZE + i];
        }
    }

    return M_transpose;
}