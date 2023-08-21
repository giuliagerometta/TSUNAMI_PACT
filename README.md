# TSUNAMI_PACT

## build
install [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
CUDA versione required >= 11.8

to compile:

```
$ git clone git@github.com:giuliagerometta/TSUNAMI_PACT.git
$ cd TSUNAMI_PACT/
$ make all
```

first you need to generate txt file containing sequences to align in the format supported by our application

```
$ cd bin/
$ ./generate_dataset num_couples sequence_length error_rate
```

within the same bin folder you can find the wfa_gpu executable

```
$ cd bin/
$ ./wfa_gpu mismatch gap_opening gap_extension file_name wf_length check
```

## legend
num_couples -> number of couples
sequence_length -> length of the sequence
error_rate -> between 1 and 100, it represents the percentage of different bases between pattern and text

mismatch -> mismatch affine penalty
gap_opening -> gap opening affine penalty
gap_extension -> gap extension affine penalty
file_name -> name of the file containing the couples of sequences
wf_length -> 
check -> 0 or 1, flag which sets the verify mode to compare TSUNAMI results with [WFA2-lib](https://github.com/smarco/WFA2-lib.git)

note that the the tool supports only match = 0
