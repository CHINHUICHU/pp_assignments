NVFLAGS := -std=c++11 -O3 -rdc=true -arch=sm_70 -Xcompiler -fopenmp
LDFLAGS := -lgomp
TARGET := hw4

.PHONY: all
all: $(TARGET)

$(TARGET): sha256.o hw4.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o hw4 hw4.cu sha256.o

sha256.o: sha256.cu
	nvcc $(NVFLAGS) -c sha256.cu

clean:
	rm -rf hw4 *.o
