CXX := g++
HIPCC := hipcc

CXXFLAGS := -std=c++17 -O3
HIPFLAGS := $(CXXFLAGS) --offload-arch=gfx908 -fopenmp
LDFLAGS := -fopenmp

TARGET := hw5
SEQUENTIAL := nbody

.PHONY: all
all: $(TARGET)

.PHONY: hw5
hw5: hw5.cpp
	$(HIPCC) $(HIPFLAGS) -o $(TARGET) hw5.cpp $(LDFLAGS)

.PHONY: seq
seq: nbody.cc
	$(CXX) $(CXXFLAGS) -fopenmp -o $(SEQUENTIAL) nbody.cc

.PHONY: clean
clean:
	rm -f $(TARGET) $(SEQUENTIAL) *.o
