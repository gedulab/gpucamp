all: gesimt

gesimt:gesimt.cu
	nvcc -Xcompiler -fopenmp -DADVDBG -G -g $^ -o $@dbg
	nvcc -Xcompiler -fopenmp -G -g $^ -o $@

clean:
	rm gesimt gesimtdbg
