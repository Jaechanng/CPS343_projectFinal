// $Smake: nvcc -arch=compute_61 -code=sm_61,sm_75,sm_80 -O3 -o %F %f
//
// add-matrices-cuda.cu - addition of two matrices on GPU device
//
// This program follows a very standard pattern:
//  1) allocate memory on host
//  2) allocate memory on device
//  3) initialize memory on host
//  4) copy memory from host to device
//  5) execute kernel(s) on device
//  6) copy result(s) from device to host
//
// Note: it may be possible to initialize memory directly on the device,
// in which case steps 3 and 4 are not necessary, and step 1 is only
// necessary to allocate memory to hold results.

#include <stdio.h>
#include <cuda.h>
#include "wtime.c"

//-----------------------------------------------------------------------------
// Kernel that executes on CUDA device

__global__ void add_matrices(
    float *c,      // out - pointer to result matrix c
    float *a,      // in  - pointer to summand matrix a
    float *b,      // in  - pointer to summand matrix b
    int m,       // in  - number of rows
    int n      // in  - number of columns
    )
{
    // Assume 2d block grid and 2-D block
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < m && col < n)
    {
        int idx = row * n + col;    
        c[idx] = a[idx] + b[idx];
    }
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Main program executes on host device

int main(int argc, char* argv[])
{
    // determine matrix dimensions
    int n = 10;      // set default number of columns
    int m = 10;      // set default number of rows
    if (argc == 3)
    {
        n = atoi(argv[1]);  // override default number of columns
        m = atoi(argv[2]);  // override default number of rows
    } else if(argc != 1)
    {
        fprintf(stderr, "Usage: %s [num_cols num_rows]\n", argv[0]);
        exit(1);
    }   

    // determine matrix size in bytes
    const size_t matrix_size = n * m * sizeof(float);

    // declare pointers to vectors in host memory and allocate memory
    float *a, *b, *c;
    a = (float*) malloc(matrix_size);
    b = (float*) malloc(matrix_size);
    c = (float*) malloc(matrix_size);

    // declare pointers to vectors in device memory and allocate memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**) &d_a, matrix_size);
    cudaMalloc((void**) &d_b, matrix_size);
    cudaMalloc((void**) &d_c, matrix_size);

    // initialize matrices and copy them to device
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            a[i * n + j] =   1.0 * (i * n + j);
            b[i * n + j] = 100.0 * (i * n + j);
        }
    }
    double time_start = wtime();
    cudaMemcpy(d_a, a, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, matrix_size, cudaMemcpyHostToDevice);

    // do calculation on device
    dim3 block_size(32, 32);
    dim3 num_blocks((n-1 + block_size.x) / block_size.x, (m-1 + block_size.y) / block_size.y);
    add_matrices<<<num_blocks, block_size>>>(d_c, d_a, d_b, m, n);

    // int block_size = 16;
    // int num_blocks = (n - 1 + block_size) / block_size;
    // add_vectors<<<num_blocks, block_size>>>(d_c, d_a, d_b, n);

    // retrieve result from device and store on host
    cudaMemcpy(c, d_c, matrix_size, cudaMemcpyDeviceToHost);
    double time_end = wtime();
    double elapsed_time = time_end - time_start;
    printf("Time to add %d x %d matrices: %f seconds\n", m, n, elapsed_time);

    // print results for matrices up to size 100x100
    if (n <= 100 && m <= 100)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                printf("%8.2f + %8.2f = %8.2f\n", a[i * n + j], b[i * n + j], c[i * n + j]);
             }
        }
    }

    // cleanup and quit
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);
  
    return 0;
}
