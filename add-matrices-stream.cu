// $Smake: nvcc -arch=compute_61 -code=sm_61,sm_75,sm_80 -O3 -o %F %f
//
// add-matrices-stream.cu - addition of two matrices on GPU device using
// CUDA streams to overlap data transfers and kernel execution.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "wtime.c"

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess)                                               \
        {                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

#define NUM_STREAMS 4
#define ROWS_PER_CHUNK 512

//-----------------------------------------------------------------------------
// Kernel that executes on CUDA device

__global__ void add_matrices(
    float *c,      // out - pointer to result matrix c
    float *a,      // in  - pointer to summand matrix a
    float *b,      // in  - pointer to summand matrix b
    int m,         // in  - number of rows in this chunk
    int n          // in  - number of columns
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

    if (n <= 0 || m <= 0)
    {
        fprintf(stderr, "Matrix dimensions must be positive.\n");
        exit(1);
    }

    // determine matrix size in bytes
    const size_t matrix_size = (size_t)n * m * sizeof(float);

    // cudaMemcpyAsync needs pinned host memory to overlap copies with kernels.
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocHost((void**) &a, matrix_size));
    CUDA_CHECK(cudaMallocHost((void**) &b, matrix_size));
    CUDA_CHECK(cudaMallocHost((void**) &c, matrix_size));

    // initialize matrices
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            a[i * n + j] =   1.0 * (i * n + j);
            b[i * n + j] = 100.0 * (i * n + j);
        }
    }

    cudaStream_t streams[NUM_STREAMS];
    float *d_a[NUM_STREAMS];
    float *d_b[NUM_STREAMS];
    float *d_c[NUM_STREAMS];

    const int rows_per_chunk = ROWS_PER_CHUNK < m ? ROWS_PER_CHUNK : m;
    const size_t chunk_size = (size_t)rows_per_chunk * n * sizeof(float);

    for (int s = 0; s < NUM_STREAMS; s++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc((void**) &d_a[s], chunk_size));
        CUDA_CHECK(cudaMalloc((void**) &d_b[s], chunk_size));
        CUDA_CHECK(cudaMalloc((void**) &d_c[s], chunk_size));
    }
    double time_start = wtime();
    // Copy chunks to the device, run kernels, and copy results back.
    // Operations within one stream stay ordered; different streams can overlap.
    dim3 block_size(32, 32);
    for (int row = 0; row < m; row += rows_per_chunk)
    {
        int stream_id = (row / rows_per_chunk) % NUM_STREAMS;
        int chunk_rows = rows_per_chunk;
        if (row + chunk_rows > m)
        {
            chunk_rows = m - row;
        }

        size_t offset = (size_t)row * n;
        size_t bytes = (size_t)chunk_rows * n * sizeof(float);

        CUDA_CHECK(cudaMemcpyAsync(d_a[stream_id], a + offset, bytes,
                                   cudaMemcpyHostToDevice, streams[stream_id]));
        CUDA_CHECK(cudaMemcpyAsync(d_b[stream_id], b + offset, bytes,
                                   cudaMemcpyHostToDevice, streams[stream_id]));

        dim3 num_blocks((n - 1 + block_size.x) / block_size.x,
                        (chunk_rows - 1 + block_size.y) / block_size.y);
        add_matrices<<<num_blocks, block_size, 0, streams[stream_id]>>>(
            d_c[stream_id], d_a[stream_id], d_b[stream_id], chunk_rows, n);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(c + offset, d_c[stream_id], bytes,
                                   cudaMemcpyDeviceToHost, streams[stream_id]));
    }

    for (int s = 0; s < NUM_STREAMS; s++)
    {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }
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
    for (int s = 0; s < NUM_STREAMS; s++)
    {
        CUDA_CHECK(cudaFree(d_a[s]));
        CUDA_CHECK(cudaFree(d_b[s]));
        CUDA_CHECK(cudaFree(d_c[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    CUDA_CHECK(cudaFreeHost(a));
    CUDA_CHECK(cudaFreeHost(b));
    CUDA_CHECK(cudaFreeHost(c));

    return 0;
}
