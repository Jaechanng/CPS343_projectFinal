/*
 * This program reads a matrix from an HDF5 file and estimates the dominant eigenvalue using the power
 * method. This version uses the CUBLAS library for the matrix-vector multiply and vector operations.
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
T dotProductOnDevice(T* x_d, T* y_d, unsigned int n)
{
    const int numThread = 128;
    const int smemSize = numThread * sizeof(T); 
    int numBlock = (n + numThread - 1) / numThread;

    T* z_d;
    cudaChkErr(cudaMalloc(&z_d, numBlock * sizeof(T)));

    dotProductInit<T><<<numBlock, numThread, smemSize>>>(x_d, y_d, z_d, n);
    cudaChkErr(cudaDeviceSynchronize());
    cudaChkErr(cudaGetLastError());
    while (numBlock > 1)
    {
        n = numBlock;
        numBlock = (n + numThread - 1) / numThread;
        reductionSumKernel<T><<<numBlock, numThread, smemSize>>>(z_d, n);
        cudaChkErr(cudaDeviceSynchronize());
        cudaChkErr(cudaGetLastError());
    }

    T sum;
    cudaChkErr(cudaMemcpy(&sum, z_d, sizeof(T), cudaMemcpyDeviceToHost));
    cudaChkErr(cudaFree(z_d));

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
void scaleOnDevice(T* x_d, T alpha, int n)
{
    const int threadsPerBlock = 256;
    const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    scaleKernel<T><<<numBlocks, threadsPerBlock>>>(x_d, alpha, n);
    cudaChkErr(cudaDeviceSynchronize());
    cudaChkErr(cudaGetLastError());
}

//----------------------------------------------------------------------------

template <typename T>
void normalizeOnDevice(T* x_d, int n)
{
    T norm2 = dotProductOnDevice<T>(x_d, x_d, n);
    T norm = sqrt(norm2);
    scaleOnDevice<T>(x_d, (T)1.0 / norm, n);
}

// Kernel for matrix-vector product: y = A * x
// Assumes A is stored in column-major order

template <typename T>
__global__ void matrixVectorProductKernel(T* A, T* x, T* y, unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
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
void matrixVectorProductOnDevice(T* A_d, T* x_d, T* y_d, unsigned int n)
{
    const int threadsPerBlock = 256;
    const int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    matrixVectorProductKernel<T><<<numBlocks, threadsPerBlock>>>(A_d, x_d, y_d, n);
    cudaChkErr(cudaDeviceSynchronize());
    cudaChkErr(cudaGetLastError());
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

    
    // allocate GPU memory
    double *d_A, *x_d, *y_d;
    cudaMalloc((void**)&d_A, n*n*sizeof(double));
    cudaMalloc((void**)&x_d, n*sizeof(double));
    cudaMalloc((void**)&y_d, n*sizeof(double));
    
    // copy matrix A to device
    cudaChkErr(cudaMemcpy(
        d_A,
        A,
        n * n * sizeof(double),
        cudaMemcpyHostToDevice
    ));
    
    // initialize x on host, then copy to device
    double* x = new double[n];
    for(int i = 0; i < n; i++) {
        x[i] = 1.0;
    }
    cudaChkErr(cudaMemcpy(
        x_d,
        x,
        n * sizeof(double),
        cudaMemcpyHostToDevice
    ));
    delete[] x;  // host x is no longer needed after this point
    
    // normalize x
    normalizeOnDevice<double>(x_d, n);
    
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
        matrixVectorProductOnDevice<double>(d_A, x_d, y_d, n); // compute the matrix-vector product on the device using our custom implementation

        // update eigenvalue estimate
        lambda_old = lambda_new;
        lambda_new = dotProductOnDevice<double>(x_d, y_d, n); // compute the dot product on the device using our custom implementation

        // compute norm of y
        double norm_y = sqrt(dotProductOnDevice<double>(y_d, y_d, n));
        
        // scale y by 1/norm_y
        scaleOnDevice<double>(y_d, 1.0 / norm_y, n);
        
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
    cudaFree(d_A);
    cudaFree(x_d);
    cudaFree(y_d);

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
