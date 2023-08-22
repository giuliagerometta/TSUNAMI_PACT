#!/bin/bash

make all

cd /bin

echo "generating dataset: 100k reads of 128 bases with 1% error rate"
./generate_dataset 100000 128 1
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 21 1

echo "generating dataset: 100k reads of 512 bases with 1% error rate"
./generate_dataset 100000 512 1
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 21 1

echo "generating dataset: 50k reads of 1024 bases with 1% error rate"
./generate_dataset 50000 1024 1
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 200 1

echo "generating dataset: 10k reads of 10240 bases with 1% error rate"
./generate_dataset 10000 10240 1
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 400 1

echo "generating dataset: 100k reads of 128 bases with 2% error rate"
./generate_dataset 100000 128 2
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 42 1

echo "generating dataset: 100k reads of 512 bases with 2% error rate"
./generate_dataset 100000 512 2
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 42 1

echo "generating dataset: 50k reads of 1024 bases with 2% error rate"
./generate_dataset 50000 1024 2
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 200 1

echo "generating dataset: 10k reads of 10240 bases with 2% error rate"
./generate_dataset 10000 10240 2
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 800 1

echo "generating dataset: 100k reads of 128 bases with 5% error rate"
./generate_dataset 100000 128 5
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 105 1

echo "generating dataset: 100k reads of 512 bases with 5% error rate"
./generate_dataset 100000 512 5
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 105 1

echo "generating dataset: 50k reads of 1024 bases with 5% error rate"
./generate_dataset 50000 1024 5
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 200 1

echo "generating dataset: 5k reads of 10240 bases with 5% error rate"
./generate_dataset 5000 10240 5
echo "aligning..."
./wfa_gpu 4 6 2 sequences.txt 2000 1