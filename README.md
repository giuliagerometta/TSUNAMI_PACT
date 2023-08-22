# TSUNAMI

## Introduction

Pairwise sequence alignment represents a fundamental step in the genome assembly pipeline, being the most time-consuming step and the bottleneck factor in multiple bioinformatics applications. Exact pairwise alignment methods like Smith-Waterman and Needleman–Wunsch, often cannot satisfy the performance required by these tools, because of their quadratic time complexity. Furthermore, given the increasing computational cost of analyzing third-generation sequences, the community is moving towards different alignment methods and hardware-accelerated solutions to overcome the limitations of these algorithms. In this scenario, we present TSUNAMI, a highly-optimized implementation of the WaveFront Alignment algorithm on GPU. TSUNAMI exploits GPUs high-parallel computing to fasten the execution of the WaveFront Alignment algorithm, a novel alignment methodology exploiting homologous regions between the target sequences. By doing so, we are able to reduce both time and space complexity in our GPU implementation. Our results show that TSUNAMI achieves improvements up to 4512.28× in terms of speedup when compared to the multi-threaded State-of-the-Art software implementation run on Intel Xeon Silver 4208 using 16 threads in total.
We also compared our design against all the recently released hardware-accelerated solutions present in the State-of-the-Art, observing speedups up to 14.81× with respect to the best performing hardware-accelerated implementation in the literature, reaching up to 42604.98 GCUPS (Giga Cell Updates Per Second) in our best configuration. TSUNAMI also provides the support for aligning very erroneous long sequences, rendering our implementation much more useful in real world scenarios. Finally, to prove the efficiency of our design, we evaluate TSUNAMI exploiting the Berkeley Roofline model and demonstrate that our implementation is near-optimal on the NVIDIA Tesla H100.

## Usage

First clone the repository by typing:

```
$ git clone https://github.com/giuliagerometta/TSUNAMI_PACT.git
$ cd TSUNAMI_PACT
$ git submodule update --init --recursive
```

TSUNAMI repository contains also the original WFA2-Lib software to check the correctness of our GPU results.

### Compilation
TSUNAMI requires C++14 and [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) >= 11.8.
To compile TSUNAMI enter the just cloned TSUNAMI repository's folder and simply type:
```
$ make all
```
This command will create a `bin\` folder that contains two different binaries, one for generating inputs used for the alignment process and the other is the binary containing TSUNAMI itself.
TSUNAMI has been tested and optimized on three platforms (NVIDIA RTX 3080, NVIDIA A100 and NVIDIA H100) but can run on any NVIDIA GPU.

To test TSUNAMI you first need to generate a set of inputs, to do so please type:
```
$ cd bin/
$ ./generate_dataset num_couples sequence_length error_rate
```
This will generate a file called `sequences.txt` which contains the input sequences for our application.
Then within the same bin folder, you can find the wfa_gpu executable which contains TSUNAMI, to use it simply type:
```
$ cd bin/
$ ./wfa_gpu mismatch gap_opening gap_extension input_file_name wf_length check
```

### Application input legend
```
generate_dataset:

num_couples -> number of couples
sequence_length -> length of the sequence
error_rate -> between 1 and 100, it represents the percentage of different bases between pattern and text
```
```
wfa_gpu:

mismatch -> mismatch affine penalty
gap_opening -> gap opening affine penalty
gap_extension -> gap extension affine penalty
input_file_name -> name of the file containing the couples of sequences
wf_length -> length of the wavefront for the alignment
check -> 0 or 1, flag which sets the verify mode to compare TSUNAMI results with [WFA2-lib](https://github.com/smarco/WFA2-lib.git)
```
### Artifact evaluation

To test our implementation using the same settings shown in our paper, please use one of the pre-defined bash scripts,
these will automatically create datasets and execute them together with comparing the attained results against software.
Please refer to the appropriate bash script depending on the GPU you want to test.
