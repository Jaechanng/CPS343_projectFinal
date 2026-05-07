/*
 * This program reads a matrix from an HDF5 file and estimates the dominant eigenvalue using the power
 * method. This version uses custom CUDA kernels and nondefault CUDA streams to overlap chunked
 * memory copies and row-block kernel launches.
*/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>   
#include <cstdlib>    
#include <cstdio>
#include <cstring>
#include <cmath>
#include "wtime.c"
#include <hdf5.h>

#include <cuda_runtime.h>
#include <cuda.h>

static const int NUM_STREAMS = 4;

// Check return values from HDF5 routines
#define CHKERR(status, name) \
    if ((status) < 0) { \
        fprintf(stderr, "Error: failure in %s\n", name); \
        exit(EXIT_FAILURE); \
    }

// Macro to index matrices in column-major (Fortran) order
#define IDX(i,j,stride) ((i)+(j)*(stride))  // column major

// Alternative macro for 1-based indexing (Fortran style)
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

// CUDA error checking macro
#if !defined(cudaChkErr)
#define cudaChkErr(x) do { cudaError_t err = (x); if(err != cudaSuccess) {\
    fprintf(stderr,"Error \"%s\" at %s:%d\n",\
            cudaGetErrorString(err),__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#endif

//----------------------------------------------------------------------------

template <typename T>
__device__ void warpReduce(volatile T* shmem, unsigned int tid)
{
    shmem[tid] += shmem[tid + 32];
    shmem[tid] += shmem[tid + 16];
    shmem[tid] += shmem[tid +  8];
    shmem[tid] += shmem[tid +  4];
    shmem[tid] += shmem[tid +  2];
    shmem[tid] += shmem[tid +  1];
}

template <typename T>
__global__ void dotProductInit(T* x, T* y, T* outData, unsigned int n)
{
    extern __shared__ T shmem[];
    const unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    shmem[threadIdx.x] = (i < n ? x[i] * y[i] : 0) +
        (i + blockDim.x < n ? x[i + blockDim.x] * y[i + blockDim.x] : 0);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        warpReduce<T>(shmem, threadIdx.x);
    }

    if (threadIdx.x == 0) outData[blockIdx.x] = shmem[0];
}

template <typename T>
__global__ void reductionSumKernel(T* data, unsigned int n)
{
    extern __shared__ T shmem[];
    const unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    shmem[threadIdx.x] = (i < n ? data[i] : 0) +
        (i + blockDim.x < n ? data[i + blockDim.x] : 0);
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        warpReduce<T>(shmem, threadIdx.x);
    }

    if (threadIdx.x == 0) data[blockIdx.x] = shmem[0];
}

template <typename T>
T dotProductOnDevice(T* x_d, T* y_d, unsigned int n, cudaStream_t stream)
{
    const int numThread = 128;
    const int smemSize = numThread * sizeof(T); 
    int numBlock = (n + numThread - 1) / numThread;

    T* z_d;
    cudaChkErr(cudaMalloc(&z_d, numBlock * sizeof(T)));

    dotProductInit<T><<<numBlock, numThread, smemSize, stream>>>(x_d, y_d, z_d, n);
    cudaChkErr(cudaStreamSynchronize(stream));
    cudaChkErr(cudaGetLastError());
    while (numBlock > 1)
    {
        n = numBlock;
        numBlock = (n + numThread - 1) / numThread;
        reductionSumKernel<T><<<numBlock, numThread, smemSize, stream>>>(z_d, n);
        cudaChkErr(cudaStreamSynchronize(stream));
        cudaChkErr(cudaGetLastError());
    }

    T sum;
    cudaChkErr(cudaMemcpyAsync(&sum, z_d, sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaChkErr(cudaStreamSynchronize(stream));
    cudaChkErr(cudaFree(z_d));

    return sum;
}

template <typename T>
T dotProductOnDevice(T* x_d, T* y_d, unsigned int n)
{
    cudaStream_t stream;
    cudaChkErr(cudaStreamCreate(&stream));
    T sum = dotProductOnDevice<T>(x_d, y_d, n, stream);
    cudaChkErr(cudaStreamDestroy(stream));
    return sum;
}

//----------------------------------------------------------------------------

template <typename T>
__global__ void scaleKernel(T* x, T alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= alpha;
    }
}

template <typename T>
void scaleOnDevice(T* x_d, T alpha, int n, cudaStream_t stream)
{
    const int threadsPerBlock = 256;
    const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    scaleKernel<T><<<numBlocks, threadsPerBlock, 0, stream>>>(x_d, alpha, n);
    cudaChkErr(cudaStreamSynchronize(stream));
    cudaChkErr(cudaGetLastError());
}

template <typename T>
void scaleOnDevice(T* x_d, T alpha, int n)
{
    cudaStream_t stream;
    cudaChkErr(cudaStreamCreate(&stream));
    scaleOnDevice<T>(x_d, alpha, n, stream);
    cudaChkErr(cudaStreamDestroy(stream));
}

//----------------------------------------------------------------------------

template <typename T>
void normalizeOnDevice(T* x_d, int n, cudaStream_t stream)
{
    T norm2 = dotProductOnDevice<T>(x_d, x_d, n, stream);
    T norm = sqrt(norm2);
    scaleOnDevice<T>(x_d, (T)1.0 / norm, n, stream);
}

template <typename T>
void normalizeOnDevice(T* x_d, int n)
{
    cudaStream_t stream;
    cudaChkErr(cudaStreamCreate(&stream));
    normalizeOnDevice<T>(x_d, n, stream);
    cudaChkErr(cudaStreamDestroy(stream));
}

// Kernel for matrix-vector product: y = A * x
// Assumes A is stored in column-major order

template <typename T>
__global__ void matrixVectorProductKernel(T* A, T* x, T* y, unsigned int n,
                                          unsigned int rowStart, unsigned int rowEnd)
{
    unsigned int i = rowStart + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < rowEnd) {
        T sum = 0;
        for (unsigned int j = 0; j < n; j++) {
            sum += A[i + j * n] * x[j];  // column-major access
        }
        y[i] = sum;
    }
}

//----------------------------------------------------------------------------

// Host function to perform matrix-vector product on device
// y = A * x, where A is n x n matrix, x and y are n-vectors

template <typename T>
void matrixVectorProductOnDevice(T* A_d, T* x_d, T* y_d, unsigned int n,
                                 cudaStream_t* streams, int numStreams)
{
    const int threadsPerBlock = 256;
    const unsigned int rowsPerStream = (n + numStreams - 1) / numStreams;
    
    // Launch independent row blocks into nondefault streams.
    for (int s = 0; s < numStreams; s++) {
        const unsigned int rowStart = s * rowsPerStream;
        const unsigned int rowEnd = rowStart + rowsPerStream < n ? rowStart + rowsPerStream : n;
        if (rowStart >= rowEnd) continue;

        const unsigned int rowCount = rowEnd - rowStart;
        const int numBlocks = (rowCount + threadsPerBlock - 1) / threadsPerBlock;
        matrixVectorProductKernel<T><<<numBlocks, threadsPerBlock, 0, streams[s]>>>(
            A_d, x_d, y_d, n, rowStart, rowEnd
        );
    }

    for (int s = 0; s < numStreams; s++) {
        cudaChkErr(cudaStreamSynchronize(streams[s]));
    }
    cudaChkErr(cudaGetLastError());
}

template <typename T>
void matrixVectorProductOnDevice(T* A_d, T* x_d, T* y_d, unsigned int n)
{
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) cudaChkErr(cudaStreamCreate(&streams[s]));
    matrixVectorProductOnDevice<T>(A_d, x_d, y_d, n, streams, NUM_STREAMS);
    for (int s = 0; s < NUM_STREAMS; s++) cudaChkErr(cudaStreamDestroy(streams[s]));
}


//----------------------------------------------------------------------------

void synchronizeStreams(cudaStream_t* streams, int numStreams)
{
    for (int s = 0; s < numStreams; s++) {
        cudaChkErr(cudaStreamSynchronize(streams[s]));
    }
}

template <typename T>
void launchCopyToDeviceAsyncChunks(T* dst_d, const T* src_h, size_t count,
                                   cudaStream_t* streams, int numStreams)
{
    const size_t elemsPerStream = (count + numStreams - 1) / numStreams;

    for (int s = 0; s < numStreams; s++) {
        const size_t offset = (size_t)s * elemsPerStream;
        if (offset >= count) continue;

        const size_t elems = offset + elemsPerStream < count ? elemsPerStream : count - offset;
        cudaChkErr(cudaMemcpyAsync(
            dst_d + offset,
            src_h + offset,
            elems * sizeof(T),
            cudaMemcpyHostToDevice,
            streams[s]
        ));
    }
}

template <typename T>
void copyToDeviceAsyncChunks(T* dst_d, const T* src_h, size_t count,
                             cudaStream_t* streams, int numStreams)
{
    launchCopyToDeviceAsyncChunks<T>(dst_d, src_h, count, streams, numStreams);
    synchronizeStreams(streams, numStreams);
}

template <typename T>
__global__ void setVectorKernel(T* x, T value, unsigned int start, unsigned int end)
{
    unsigned int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < end) {
        x[i] = value;
    }
}

template <typename T>
void launchSetVectorChunks(T* x_d, T value, unsigned int n,
                           cudaStream_t* streams, int numStreams)
{
    const int threadsPerBlock = 256;
    const unsigned int elemsPerStream = (n + numStreams - 1) / numStreams;

    for (int s = 0; s < numStreams; s++) {
        const unsigned int start = s * elemsPerStream;
        const unsigned int end = start + elemsPerStream < n ? start + elemsPerStream : n;
        if (start >= end) continue;

        const unsigned int elemCount = end - start;
        const int numBlocks = (elemCount + threadsPerBlock - 1) / threadsPerBlock;
        setVectorKernel<T><<<numBlocks, threadsPerBlock, 0, streams[s]>>>(x_d, value, start, end);
    }
}

//----------------------------------------------------------------------------

void  read_matrix_hdf5(char* fname, const char* name, double** A, int& n) {
    // read the matrix from the file and store it in column-major order
    hid_t file_id, dataset_id, file_dataspace_id;
    herr_t status;
    hsize_t dims[2];
    int rank;
    int ndims;
    hsize_t num_elem;

    //open the file
    file_id = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
    CHKERR(file_id, "H5Fopen");

    //open the dataset
    dataset_id = H5Dopen(file_id, name, H5P_DEFAULT);
    CHKERR(dataset_id, "H5Dopen");

    //dermine the dimensions of the dataset
    file_dataspace_id = H5Dget_space(dataset_id);
    CHKERR(file_dataspace_id, "H5Dget_space");
    rank  = H5Sget_simple_extent_ndims(file_dataspace_id);
    ndims = H5Sget_simple_extent_dims(file_dataspace_id, dims, nullptr);
    if (rank != 2) {
        fprintf(stderr, "Error: dataset is not 2-dimensional\n");
        exit(EXIT_FAILURE);
    }
    if(ndims < 0) {
        fprintf(stderr, "Error: unable to determine the dimensions of the dataset\n");
        exit(EXIT_FAILURE);
    }

    
    //allocate memory for the matrix
    num_elem = H5Sget_simple_extent_npoints(file_dataspace_id);
    *A = new double[num_elem];
    if(dims[0] != dims[1]) {
        fprintf(stderr, "Error: dataset is not square\n");
        exit(EXIT_FAILURE);
    }
    n = (int)dims[0];

    //read the dataset into the matrix
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, *A); 
    CHKERR(status, "H5Dread");

    //close resources
    status = H5Sclose(file_dataspace_id); CHKERR(status, "H5Sclose");
    status = H5Dclose(dataset_id); CHKERR(status, "H5Dclose");
    status = H5Fclose(file_id); CHKERR(status, "H5Fclose");
}

void power_method(double* A, int n, double tol, int max_iter, double* lambda, int* iters, double* iter_time, int verbose) {

    const int numStreams = n < NUM_STREAMS ? n : NUM_STREAMS;
    cudaStream_t streams[NUM_STREAMS];
    for (int s = 0; s < numStreams; s++) {
        cudaChkErr(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
    }
    
    // allocate GPU memory
    double *d_A, *x_d, *y_d;
    const size_t matrixElems = (size_t)n * (size_t)n;
    cudaChkErr(cudaMalloc((void**)&d_A, matrixElems * sizeof(double)));
    cudaChkErr(cudaMalloc((void**)&x_d, (size_t)n * sizeof(double)));
    cudaChkErr(cudaMalloc((void**)&y_d, (size_t)n * sizeof(double)));
    
    cudaError_t registerStatus = cudaHostRegister(A, matrixElems * sizeof(double), cudaHostRegisterPortable);
    if (registerStatus != cudaSuccess) {
        cudaGetLastError();
    }

    // Copy matrix chunks and enqueue x initialization kernels in the same nondefault streams.
    // This lets setup kernels from early streams overlap with copies from later streams.
    launchCopyToDeviceAsyncChunks<double>(d_A, A, matrixElems, streams, numStreams);
    launchSetVectorChunks<double>(x_d, 1.0, n, streams, numStreams);
    synchronizeStreams(streams, numStreams);
    cudaChkErr(cudaGetLastError());
    
    // normalize x
    normalizeOnDevice<double>(x_d, n, streams[0]);
    
    // initialize values for eigenvalue tracking
    double lambda_new = 0.0;
    double lambda_old = lambda_new + 2.0 * tol;
    double delta = fabs(lambda_new - lambda_old);
    int iter = 0;

    // start the timer
    double start_time = wtime();
    
    while(delta >= tol && iter <= max_iter) {
        iter++;
        
        // compute y = A*x
        matrixVectorProductOnDevice<double>(d_A, x_d, y_d, n, streams, numStreams); // compute row chunks asynchronously using CUDA streams

        // update eigenvalue estimate
        lambda_old = lambda_new;
        lambda_new = dotProductOnDevice<double>(x_d, y_d, n, streams[0]); // compute the dot product on the device using our custom implementation

        // compute norm of y
        double norm_y = sqrt(dotProductOnDevice<double>(y_d, y_d, n, streams[0]));
        
        // scale y by 1/norm_y
        scaleOnDevice<double>(y_d, 1.0 / norm_y, n, streams[0]);
        
        // swap pointers to avoid copying
        double* temp = x_d;
        x_d = y_d;
        y_d = temp;

        delta = fabs(lambda_new - lambda_old);
        if (verbose) {
            printf("%3d: lambda = %12.9f, delta = %.4e\n", iter, lambda_new, delta);
        }
    }
    
    // stop the timer
    double end_time = wtime();

    // report results
    *iter_time = end_time - start_time;
    *lambda = lambda_new;
    *iters = iter;

    // free device memory
    cudaChkErr(cudaFree(d_A));
    cudaChkErr(cudaFree(x_d));
    cudaChkErr(cudaFree(y_d));
    if (registerStatus == cudaSuccess) {
        cudaChkErr(cudaHostUnregister(A));
    }
    for (int s = 0; s < numStreams; s++) {
        cudaChkErr(cudaStreamDestroy(streams[s]));
    }

}

int usage(char *progname) {
    fprintf(stderr, "Usage: %s [-q] [-v] [-e tol] [-m maxiter] filename\n", progname);
    return EXIT_FAILURE;
}



int main(int argc, char *argv[]) {
    // read the matrix from the file
    // estimate the dominant eigenvalue using the power method
    // output the results
    double* A; // pointer to the matrix stored in column-major order
    int n; // number of rows in the matrix
    
    char* filename = NULL; // name of the file containing the matrix
    int verbose = 0; // flag for verbose output
    double tol = 1e-6; // tolerance for convergence
    int max_iter = 1000; // maximum number of iterations
    int quiet = 0; // flag for quiet output

    //parse the command line arguments
    int ch;
    while((ch = getopt(argc, argv, "e:m:qv")) != -1) {
        switch(ch) {
            case 'e':
                tol = atof(optarg);
                break;
            case 'm':
                max_iter = atoi(optarg);
                break;
            case 'q':
                quiet = 1;
                break;  
            case 'v':
                verbose = 1;
                break;
            default:
                usage(argv[0]);
                return EXIT_FAILURE;
        }
    }
    
    if (optind >= argc) {
    usage(argv[0]);
    return EXIT_FAILURE;
    }
    filename = argv[optind];

    double start_time = wtime();
    read_matrix_hdf5(filename, "/A/value", &A, n);
    double end_time = wtime();
    double read_time = end_time - start_time;

    // //print matrix
    // printf("Matrix A:\n");
    // dumpMatrix(A, n, n, n);

    //call the power method
    //this is where I need to call the kernel function 
    double lambda;
    int iters;
    double compute_time;
    power_method(A, n, tol, max_iter, &lambda, &iters, &compute_time, verbose);

    //report
    //handle non-convergence
    if (iters > max_iter) {
        fprintf(stderr, "*** WARNING ****: maximum number of iterations exceeded\n");
    }

    if (quiet) {
        printf("%5d x %5d %5d %10.6f %10.6f %10.6f\n", n, n, iters, lambda, read_time, compute_time);
    } else {
        printf("matrix dimensions: %d x %d; tolerance: %8.4e; max iterations: %d\n",
            n, n, tol, max_iter);
        printf("elapsed HDF5 read time = %10.6f seconds\n", read_time);
        printf("elapsed compute time   = %10.6f seconds\n", compute_time);
        printf("eigenvalue = %.6f found in %d iterations\n", lambda, iters);
    }

    delete[] A;
}
