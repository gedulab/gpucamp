all: gemm2

gemm2:gemm2.cu
	nvcc -Xcompiler -fopenmp -G -g $^ -o $@dbg
	nvcc -Xcompiler -fopenmp -G -g $^ -o $@

clean:
	rm gemm2 gemm2dbg
