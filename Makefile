FOLDER_WFA=WFA2-lib/
FOLDER_BIN=bin
COMPILER_GPU = nvcc
COMPILER = g++
CFLAGS=-std=c++14 -O3

smarco:
	$(MAKE) --directory=WFA2-lib clean all

cpu: *.cpp
	$(COMPILER) $(CFLAGS) -I$(FOLDER_WFA) wfa.cpp -o $(FOLDER_BIN)/wfa -LWFA2-lib/lib/ -lwfa -fopenmp
	$(COMPILER) $(CFLAGS) -I$(FOLDER_WFA) generate_dataset.cpp -o $(FOLDER_BIN)/generate_dataset -LWFA2-lib/lib/ -lwfa -fopenmp -O3
		
gpu: *.cu 
	$(COMPILER_GPU) $(CFLAGS) -I$(FOLDER_WFA) wfa_gpu.cu -o $(FOLDER_BIN)/wfa_gpu -LWFA2-lib/lib/ -lwfa -Xcompiler -fopenmp -O3
	$(COMPILER_GPU) $(CFLAGS) -I$(FOLDER_WFA) wfa_gpu_bt.cu -o $(FOLDER_BIN)/wfa_gpu_bt -LWFA2-lib/lib/ -lwfa -Xcompiler -fopenmp -O3
	$(COMPILER_GPU) $(CFLAGS) -I$(FOLDER_WFA) wfa_gpu_compactbt.cu -o $(FOLDER_BIN)/wfa_gpu_compactbt -LWFA2-lib/lib/ -lwfa -Xcompiler -fopenmp -O3


all: smarco cpu gpu

clean: 
	rm -rf $(FOLDER_BIN)/wfa_gpu
	rm -rf $(FOLDER_BIN)/wfa_gpu_bt
	rm -rf $(FOLDER_BIN)/generate_dataset
	rm -rf $(FOLDER_BIN)/wfa
