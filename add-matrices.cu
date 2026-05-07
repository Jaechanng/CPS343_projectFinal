// $Smake: nvcc -arch=compute_61 -code=sm_61,sm_75,sm_80 -O3 -o %F %f
//
// add-matrices.cu - addition of two matrices on GPU device
//
// CPS343 final presentation version:
// Demonstrates the normal sequential CUDA workflow and compares it with
// a CUDA streams workflow that overlaps memory transfer and kernel work.
//
// Build:
//   nvcc -arch=compute_61 -code=sm_61,sm_75,sm_80 -O3 -o add-matrices add-matrices.cu
//
// Run examples:
//   ./add-matrices
//   ./add-matrices 4096 4096
//   ./add-matrices 8192 8192 4 1024
//
// Profile examples:
//   nvprof ./add-matrices 8192 8192 4 1024
//   nsys profile -o add-matrices-streams ./add-matrices 8192 8192 4 1024

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

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

//-----------------------------------------------------------------------------
// Kernel that executes on CUDA device

__global__ void add_matrices(
    float *c,    // out - pointer to result matrix c
    float *a,    // in  - pointer to summand matrix a
    float *b,    // in  - pointer to summand matrix b
    int m,       // in  - number of rows in this chunk
    int n        // in  - number of columns
    )
{
    // Assume 2-D block grid and 2-D block.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        int idx = row * n + col;
        c[idx] = a[idx] + b[idx];
    }
}

static void initialize_matrices(float *a, float *b, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            a[i * n + j] = 1.0f * (i * n + j);
            b[i * n + j] = 100.0f * (i * n + j);
        }
    }
}

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static float run_sequential(float *c, const float *a, const float *b, int m, int n)
{
    const size_t matrix_size = (size_t)m * n * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaMalloc((void **)&d_a, matrix_size));
    CUDA_CHECK(cudaMalloc((void **)&d_b, matrix_size));
    CUDA_CHECK(cudaMalloc((void **)&d_c, matrix_size));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block_size(32, 32);
    dim3 num_blocks((n - 1 + block_size.x) / block_size.x,
                    (m - 1 + block_size.y) / block_size.y);

    CUDA_CHECK(cudaEventRecord(start));

    // Normal CUDA order: copy input, run kernel, copy result back.
    CUDA_CHECK(cudaMemcpy(d_a, a, matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, matrix_size, cudaMemcpyHostToDevice));
    add_matrices<<<num_blocks, block_size>>>(d_c, d_a, d_b, m, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c, d_c, matrix_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = elapsed_ms(start, stop);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return ms;
}

static float run_streamed(float *c, const float *a, const float *b,
                          int m, int n, int num_streams, int rows_per_chunk)
{
    const size_t chunk_size = (size_t)rows_per_chunk * n * sizeof(float);
    cudaStream_t *streams = (cudaStream_t *)malloc(num_streams * sizeof(cudaStream_t));
    float **d_a = (float **)malloc(num_streams * sizeof(float *));
    float **d_b = (float **)malloc(num_streams * sizeof(float *));
    float **d_c = (float **)malloc(num_streams * sizeof(float *));
    cudaEvent_t start, stop;

    if (!streams || !d_a || !d_b || !d_c)
    {
        fprintf(stderr, "Host allocation failed\n");
        exit(1);
    }

    for (int s = 0; s < num_streams; s++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        CUDA_CHECK(cudaMalloc((void **)&d_a[s], chunk_size));
        CUDA_CHECK(cudaMalloc((void **)&d_b[s], chunk_size));
        CUDA_CHECK(cudaMalloc((void **)&d_c[s], chunk_size));
    }

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    dim3 block_size(32, 32);

    // Each chunk is ordered inside its stream:
    // async copy H->D, kernel, async copy D->H.
    // Different non-default streams can overlap with each other.
    for (int row = 0; row < m; row += rows_per_chunk)
    {
        int stream_id = (row / rows_per_chunk) % num_streams;
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

    for (int s = 0; s < num_streams; s++)
    {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = elapsed_ms(start, stop);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    for (int s = 0; s < num_streams; s++)
    {
        CUDA_CHECK(cudaFree(d_a[s]));
        CUDA_CHECK(cudaFree(d_b[s]));
        CUDA_CHECK(cudaFree(d_c[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }

    free(streams);
    free(d_a);
    free(d_b);
    free(d_c);

    return ms;
}

static int verify_result(const float *a, const float *b, const float *c, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int idx = i * n + j;
            float expected = a[idx] + b[idx];
            if (fabsf(c[idx] - expected) > 0.001f)
            {
                fprintf(stderr, "Mismatch at (%d, %d): got %.2f expected %.2f\n",
                        i, j, c[idx], expected);
                return 0;
            }
        }
    }
    return 1;
}

//-----------------------------------------------------------------------------
// Main program executes on host device

int main(int argc, char *argv[])
{
    int n = 10;              // default number of columns
    int m = 10;              // default number of rows
    int num_streams = 4;     // default number of non-default streams
    int rows_per_chunk = 512;

    if (argc == 3 || argc == 5)
    {
        n = atoi(argv[1]);
        m = atoi(argv[2]);
        if (argc == 5)
        {
            num_streams = atoi(argv[3]);
            rows_per_chunk = atoi(argv[4]);
        }
    }
    else if (argc != 1)
    {
        fprintf(stderr, "Usage: %s [num_cols num_rows [num_streams rows_per_chunk]]\n", argv[0]);
        exit(1);
    }

    if (n <= 0 || m <= 0 || num_streams <= 0 || rows_per_chunk <= 0)
    {
        fprintf(stderr, "All dimensions and stream settings must be positive.\n");
        exit(1);
    }

    const size_t matrix_size = (size_t)n * m * sizeof(float);

    // cudaMemcpyAsync needs pinned host memory to overlap copies with kernels.
    float *a, *b, *c_sequential, *c_streamed;
    CUDA_CHECK(cudaMallocHost((void **)&a, matrix_size));
    CUDA_CHECK(cudaMallocHost((void **)&b, matrix_size));
    CUDA_CHECK(cudaMallocHost((void **)&c_sequential, matrix_size));
    CUDA_CHECK(cudaMallocHost((void **)&c_streamed, matrix_size));

    initialize_matrices(a, b, m, n);

    printf("Matrix size: %d rows x %d cols (%.2f MB per matrix)\n",
           m, n, matrix_size / (1024.0 * 1024.0));
    printf("Streams demo: %d streams, %d rows per chunk\n\n",
           num_streams, rows_per_chunk);

    float sequential_ms = run_sequential(c_sequential, a, b, m, n);
    float streamed_ms = run_streamed(c_streamed, a, b, m, n, num_streams, rows_per_chunk);

    int sequential_ok = verify_result(a, b, c_sequential, m, n);
    int streamed_ok = verify_result(a, b, c_streamed, m, n);

    printf("Sequential memcpy + kernel + memcpy: %.3f ms [%s]\n",
           sequential_ms, sequential_ok ? "OK" : "FAILED");
    printf("Streamed async chunk workflow:       %.3f ms [%s]\n",
           streamed_ms, streamed_ok ? "OK" : "FAILED");

    if (streamed_ms > 0.0f)
    {
        printf("Speedup: %.2fx\n", sequential_ms / streamed_ms);
    }

    if (n <= 100 && m <= 100)
    {
        printf("\nSample results:\n");
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int idx = i * n + j;
                printf("%8.2f + %8.2f = %8.2f\n", a[idx], b[idx], c_streamed[idx]);
            }
        }
    }

    CUDA_CHECK(cudaFreeHost(a));
    CUDA_CHECK(cudaFreeHost(b));
    CUDA_CHECK(cudaFreeHost(c_sequential));
    CUDA_CHECK(cudaFreeHost(c_streamed));

    return (sequential_ok && streamed_ok) ? 0 : 1;
}
