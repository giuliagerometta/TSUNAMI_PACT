#!/bin/bash

make all

cd /bin

echo "generating dataset: 500k reads of 128 bases with 1% error rate"
./generate_dataset 500000 128 1
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 21 1

echo "generating dataset: 500k reads of 512 bases with 1% error rate"
./generate_dataset 500000 512 1
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 21 1

echo "generating dataset: 50k reads of 1024 bases with 1% error rate"
./generate_dataset 50000 1024 1
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 200 1

echo "generating dataset: 10k reads of 10240 bases with 1% error rate"
./generate_dataset 10000 10240 1
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 350 1

echo "generating dataset: 500k reads of 128 bases with 2% error rate"
./generate_dataset 500000 128 2
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 42 1

echo "generating dataset: 500k reads of 512 bases with 2% error rate"
./generate_dataset 500000 512 2
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 42 1

echo "generating dataset: 50k reads of 1024 bases with 2% error rate"
./generate_dataset 50000 1024 2
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 250 1

echo "generating dataset: 10k reads of 10240 bases with 2% error rate"
./generate_dataset 10000 10240 2
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 700 1

echo "generating dataset: 500k reads of 128 bases with 5% error rate"
./generate_dataset 500000 128 5
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 105 1

echo "generating dataset: 500k reads of 512 bases with 5% error rate"
./generate_dataset 500000 512 5
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 105 1

echo "generating dataset: 50k reads of 1024 bases with 5% error rate"
./generate_dataset 50000 1024 5
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 250 1

echo "generating dataset: 10k reads of 10240 bases with 5% error rate"
./generate_dataset 10000 10240 5
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 2000 1