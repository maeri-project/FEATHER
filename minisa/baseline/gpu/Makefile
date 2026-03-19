# GEMM Kernel Profiler Makefile

# CUDA compiler
NVCC := /usr/local/cuda/bin/nvcc

# Detect GPU architecture (default to sm_89 for RTX 4090, sm_120 for RTX 5090)
GPU_ARCH ?= sm_120

# Compiler flags
NVCC_FLAGS := -O3 -std=c++17 -arch=$(GPU_ARCH) --use_fast_math
NVCC_FLAGS += -Xcompiler -Wall

# Include paths
INCLUDES := -I.

# Source files
SOURCES := profiler.cu
TARGET := gemm_profiler

# Default target
all: $(TARGET)

$(TARGET): $(SOURCES) kernels/*.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(SOURCES)

# Run profiler
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# View results
results:
	@if [ -f gemm_profiling_results.csv ]; then \
		column -t -s',' gemm_profiling_results.csv | head -20; \
	else \
		echo "No results file found. Run 'make run' first."; \
	fi

# Help
help:
	@echo "GEMM Kernel Profiler"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build the profiler (default)"
	@echo "  run      - Build and run the profiler"
	@echo "  clean    - Remove build artifacts"
	@echo "  results  - Display profiling results"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  GPU_ARCH - Set GPU architecture (default: sm_120)"
	@echo "             Example: make GPU_ARCH=sm_89"

.PHONY: all run clean results help
