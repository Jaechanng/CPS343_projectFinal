NVCC = nvcc
NVCCFLAGS = -lineinfo -arch=compute_61 -code=sm_61,sm_75,sm_80 -O3
HDF5FLAGS ?= -lhdf5

TARGETS = dot-matrices_cuda dot-matrices_cuda_stream

.PHONY: all clean

all: $(TARGETS)

dot-matrices_cuda: dot-matrices_cuda.cu wtime.c
	$(NVCC) $(NVCCFLAGS) -o $@ dot-matrices_cuda.cu $(HDF5FLAGS)

dot-matrices_cuda_stream: dot-matrices_cuda_stream.cu wtime.c
	$(NVCC) $(NVCCFLAGS) -o $@ dot-matrices_cuda_stream.cu $(HDF5FLAGS)

clean:
	rm -f $(TARGETS)
