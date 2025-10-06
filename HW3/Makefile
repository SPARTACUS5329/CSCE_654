CXX       = CC
NVCC      = nvcc
CXXFLAGS  = -O3 -std=c++17 -Iinclude -DENABLE_CUDA
NVCCFLAGS = -O3 -std=c++17 -Iinclude -DENABLE_CUDA -ccbin=$(CXX)

CPU_SRCS  = src/main.cpp src/summa_cpu.cpp src/local_dgemm_cpu.cpp src/verify.cpp
GPU_SRCS  = src/summa_gpu.cu src/local_dgemm_gpu.cu

CPU_OBJS  = $(CPU_SRCS:.cpp=.o)
GPU_OBJS  = $(GPU_SRCS:.cu=.o)

TARGET    = summa

all: $(TARGET)

$(TARGET): $(CPU_OBJS) $(GPU_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ -L$(CUDA_HOME)/lib64 -lcudart

cpu: $(CPU_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) src/*.o
