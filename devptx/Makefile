all: geinc



geinc:
	nvcc --device-c geinc.ptx
	nvcc --device-c gemain.cu
	nvcc --device-link geinc.o gemain.o -o link.o
	nvcc --lib -o libgpu.a gemain.o geinc.o link.o
	nvcc -c geinc.cu
	g++ gemain.o geinc.o -L. -lgpu -o $@  -lcudadevrt -lcudart -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl

clean:
	rm *.o geinc

