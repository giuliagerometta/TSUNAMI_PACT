#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <fstream>
#include <ostream>
#include <algorithm> // std::random_shuffle

int main(int argc, char *argv[])
{

    if (argc < 4)
    {
        std::cout << "Required arguments: number of couples, sequences length, error rate" << std::endl;
        exit(1);
    }

    int num = atoi(argv[1]);
    int seq_size = atoi(argv[2]);
    unsigned int error_rate = atoi(argv[3]);
    unsigned int num_errors = ((seq_size * error_rate) / 100);
    std::vector<int> random_position(seq_size);
    char alphabet[4] = {'A', 'C', 'G', 'T'};
    char *pattern = (char *)malloc(sizeof(char) * seq_size);
    char *text = (char *)malloc(sizeof(char) * seq_size);
    FILE *seq_file = fopen("sequences.txt", "w");

    for (int i = 0; i < seq_size; random_position[i] = i, i++);

    std::random_shuffle(random_position.begin(), random_position.end());

    int seed = time(NULL);
    // int seed=1639942415;
    // std::cout<<"Seed: "<<seed<<std::endl;
    srand(seed);

    fprintf(seq_file, "%d\n", num);
    fprintf(seq_file, "%d\n", seq_size);
    fprintf(seq_file, "%d\n", error_rate);

    unsigned ran_idx = 0;

    for (int n = 0; n < num; n++)
    {

        for (int j = 0; j < seq_size; j++)
        {
            text[j] = alphabet[rand() % 4];
            pattern[j] = text[j];
        }

        // insert errors

#pragma omp parallel for
        for (int j = 0; j < num_errors; j++)
        {
            int idx = random_position[ran_idx];
            ran_idx++;
            ran_idx = ran_idx % seq_size;
            int old = pattern[idx];
            pattern[idx] = alphabet[rand() % 4];
            while (pattern[idx] == old)
                pattern[idx] = alphabet[rand() % 4];
        }

        fprintf(seq_file, "%s\n", pattern);
        fprintf(seq_file, "%s\n", text);
    }

    fclose(seq_file);

    return 0;
}