#include <cstdlib>
#include <string>
#include <vector>
#include <cstring>
#include "common.h"
#include <omp.h>
#include <chrono>

extern "C" {
	#include "wavefront/wavefront_align.h"
}

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define NOW std::chrono::high_resolution_clock::now();
#define NUM_THREADS 32

__global__ void wfa_dispatcher(char *pattern_d, char* text_d, int16_t *score_d, int16_t *mwavefronts_d, 
                                int16_t *iwavefronts_d, int16_t* dwavefronts_d, int max_op, int num_couples, 
                                bool *nullm_d, bool *nulli_d, bool *nulld_d, int16_t *limitsm_d, int16_t *limitsi_d, int16_t *limitsd_d,
                                int pattern_len, int text_len, int num_wavefronts, int wf_length, int max_score, int16_t mismatch, 
                                int16_t gap_opening, int16_t gap_extension, int hi, int lo, char *operations);

__device__ int count_chars(char *pattern, char *text, int seq_len);

int main(int argc, char const *argv[]){

    if(argc!=6){
		printf("ERROR! Please specify in order: mismatch, gap_opening, gap_extension, file name, check\n");
		return 0;
	}

    int num_couples;
    bool check;
    char *pattern, *text, *pattern_d, *text_d;
    int16_t *mwavefronts_d, *iwavefronts_d, *dwavefronts_d;
    bool *nullm_d, *nulli_d, *nulld_d;
    int16_t *limitsm_d, *limitsi_d, *limitsd_d;
    int16_t *score_d, *score;
    char *operations_d, *operations;

	int16_t mismatch = atoi(argv[1]);
	int16_t gap_opening = atoi(argv[2]);
	int16_t gap_extension = atoi(argv[3]);
    FILE* fp = fopen(argv[4], "r");
    check = atoi(argv[5]);

    if(fp==NULL){
		printf("ERROR: Cannot open file.\n");
		return 1;
	}

    int seq_len;
    fscanf(fp, "%d %d", &num_couples, &seq_len);
    int pattern_len = seq_len;
    int text_len = seq_len;

    score = (int16_t*)malloc(sizeof(int16_t)*num_couples);
    pattern = (char*)malloc(sizeof(char)*pattern_len*num_couples);
    text = (char*)malloc(sizeof(char)*text_len*num_couples);
    for(int i = 0; i<num_couples; i++){
		fscanf(fp, "%s", pattern+(i*pattern_len));
		fscanf(fp, "%s", text+(i*text_len));
	}

    int max_score = MAX(gap_opening + gap_extension, mismatch) + 1;
    int abs_seq_diff = ABS(pattern_len - text_len);
    int max_score_misms = MIN(pattern_len, text_len) * mismatch;
    int max_score_indel = gap_opening + abs_seq_diff * gap_extension;
    int num_wavefronts = max_score_misms + max_score_indel + 1;
    int max_op = 2 * (pattern_len + text_len);
    int hi = 0;
    int lo = 0;
    int eff_lo = lo - (max_score + 1);
    int eff_hi = hi + (max_score + 1);
    lo = MIN(eff_lo, 0);
    hi = MAX(eff_hi, 0);
    int wf_length = hi - lo + 1;
    operations = (char*)malloc(sizeof(char)*num_couples*max_op);
    
    CHECK(cudaDeviceReset());
    CHECK(cudaMalloc(&pattern_d, sizeof(char)*pattern_len*num_couples));
    CHECK(cudaMalloc(&text_d, sizeof(char)*text_len*num_couples));
    CHECK(cudaMalloc(&score_d, sizeof(int16_t)*num_couples));
    CHECK(cudaMalloc(&mwavefronts_d, sizeof(int16_t)*num_wavefronts*wf_length*num_couples));
    CHECK(cudaMalloc(&dwavefronts_d, sizeof(int16_t)*num_wavefronts*wf_length*num_couples));
    CHECK(cudaMalloc(&iwavefronts_d, sizeof(int16_t)*num_wavefronts*wf_length*num_couples));
    CHECK(cudaMalloc(&nullm_d, sizeof(bool)*num_wavefronts*num_couples));
    CHECK(cudaMalloc(&nulli_d, sizeof(bool)*num_wavefronts*num_couples));
    CHECK(cudaMalloc(&nulld_d, sizeof(bool)*num_wavefronts*num_couples));
    CHECK(cudaMalloc(&limitsm_d, sizeof(int16_t)*num_wavefronts*4*num_couples));
    CHECK(cudaMalloc(&limitsi_d, sizeof(int16_t)*num_wavefronts*4*num_couples));
    CHECK(cudaMalloc(&limitsd_d, sizeof(int16_t)*num_wavefronts*4*num_couples));
    CHECK(cudaMalloc(&operations_d, sizeof(char)*max_op*num_couples));

    CHECK(cudaMemcpy(pattern_d, pattern, sizeof(char)*num_couples*pattern_len, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(text_d, text, sizeof(char)*num_couples*text_len, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(num_couples, 1, 1);
	dim3 threadsPerBlock(NUM_THREADS, 1, 1);

    std::chrono::high_resolution_clock::time_point start = NOW;

    wfa_dispatcher<<<blocksPerGrid, threadsPerBlock>>>(pattern_d, text_d, score_d, mwavefronts_d, iwavefronts_d, 
                    dwavefronts_d, max_op, num_couples, nullm_d, nulli_d, nulld_d, limitsm_d, limitsi_d, limitsd_d,
                    pattern_len, text_len, num_wavefronts, wf_length, max_score, mismatch, gap_opening, gap_extension, 
                    hi, lo, operations_d);
    
    CHECK_KERNELCALL();
    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point end = NOW;
    std::chrono::duration<double> time_temp = (end - start);
    
    CHECK(cudaMemcpy(score, score_d, sizeof(int16_t)*num_couples, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(operations, operations_d, sizeof(char)*num_couples*max_op, cudaMemcpyDeviceToHost));

    double gcups = (pattern_len*text_len*num_couples);
    gcups/=(time_temp.count()*1E9);
    printf("Time: %lf\n", time_temp.count());
    printf("Estimated GCUPS gpu: %lf\n", gcups);

    CHECK(cudaFree(pattern_d));
    CHECK(cudaFree(text_d));
    CHECK(cudaFree(score_d));
    CHECK(cudaFree(mwavefronts_d));
    CHECK(cudaFree(iwavefronts_d));
    CHECK(cudaFree(dwavefronts_d));
    CHECK(cudaFree(nullm_d));
    CHECK(cudaFree(nulli_d));
    CHECK(cudaFree(nulld_d));
    CHECK(cudaFree(limitsm_d));
    CHECK(cudaFree(limitsi_d));
    CHECK(cudaFree(limitsd_d));

    if(check){  
        std::chrono::high_resolution_clock::time_point start_cpu = NOW;      
		// Configure alignment attributes
        int16_t *smarco_score = (int16_t*)malloc(sizeof(int16_t)*num_couples);
        #pragma omp parallel for
        for(int i=0; i<num_couples; i++){
            wavefront_aligner_attr_t attributes = wavefront_aligner_attr_default;
            attributes.distance_metric = gap_affine;
            attributes.affine_penalties.mismatch = mismatch;
            attributes.affine_penalties.gap_opening = gap_opening;
            attributes.affine_penalties.gap_extension = gap_extension;
            // Initialize Wavefront Aligner
            wavefront_aligner_t* const wf_aligner = wavefront_aligner_new(&attributes);
            
            wavefront_align(wf_aligner,pattern+(i*pattern_len),pattern_len,text+(i*text_len),text_len); // Align
            smarco_score[i] = -(wf_aligner->cigar->score);
        }
        
        std::chrono::high_resolution_clock::time_point end_cpu = NOW;
        std::chrono::duration<double> time_temp_cpu = (end_cpu - start_cpu);
        double gcups_cpu = (pattern_len*text_len*num_couples/(time_temp_cpu.count()*1000000000));
        printf("Estimated GCUPS sw: %lf\n", gcups_cpu);

        bool equal = true;
        for(int i=0; i<num_couples && equal; i++){  
            if(smarco_score[i] != score[i])
                equal = false;
        }

        if(equal) printf("\n\n                    |ALIGNEMENT FINISHED CORRECTLY|\n");
        else printf("\n\n |ERROR|\n");  
    }

    return 0;
}


__global__ void wfa_dispatcher(char *pattern_d, char* text_d, int16_t *score_d, int16_t *mwavefronts_d, 
                                int16_t *iwavefronts_d, int16_t* dwavefronts_d, int max_op, int num_couples, 
                                bool *nullm_d, bool *nulli_d, bool *nulld_d, int16_t *limitsm_d, int16_t *limitsi_d, int16_t *limitsd_d,
                                int pattern_len, int text_len, int num_wavefronts, int wf_length, int max_score, int16_t mismatch, 
                                int16_t gap_opening, int16_t gap_extension, int hi, int lo, char* operations_d){

    char *pattern = &pattern_d[pattern_len*blockIdx.x];
    char *text = &text_d[text_len*blockIdx.x];
    int16_t *mwavefronts = &mwavefronts_d[num_wavefronts*wf_length*blockIdx.x];
    int16_t *iwavefronts = &iwavefronts_d[num_wavefronts*wf_length*blockIdx.x];
    int16_t *dwavefronts = &dwavefronts_d[num_wavefronts*wf_length*blockIdx.x];
    bool *nullm = &nullm_d[num_wavefronts*blockIdx.x]; 
    bool *nulli = &nulli_d[num_wavefronts*blockIdx.x]; 
    bool *nulld = &nulld_d[num_wavefronts*blockIdx.x];
    int16_t *limitsm = &limitsm_d[4*num_wavefronts*blockIdx.x]; //lo = 0, hi = 1, wf_elements_init_min = 2,  wf_elements_init_max = 3
    int16_t *limitsi = &limitsi_d[4*num_wavefronts*blockIdx.x];
    int16_t *limitsd = &limitsd_d[4*num_wavefronts*blockIdx.x];
    char *operations = &operations_d[max_op*blockIdx.x];

    int s = 0;
    int16_t num_null_steps = 0;
    int16_t historic_max_hi = 0;
    int16_t historic_min_lo = 0;
    bool finish = false;
    int16_t alignment_k = text_len - pattern_len;
    int16_t alignment_offset = text_len;
    matrix_type component_end = matrix_M;
    int16_t matr_idx = s*wf_length + wf_length/2;
    int16_t limits_idx = s*4;

    //wavefronts initialization
    for(int i = 0; i < wf_length*num_wavefronts; i += blockDim.x){
        if(i + threadIdx.x < wf_length*num_wavefronts){
            mwavefronts[i + threadIdx.x] = OFFSET_NULL;
            iwavefronts[i + threadIdx.x] = OFFSET_NULL;
            dwavefronts[i + threadIdx.x] = OFFSET_NULL;
        }
    }
    __syncthreads();

    for(int i = 0; i < num_wavefronts; i += blockDim.x){
        if(i + threadIdx.x < num_wavefronts){
            nullm[i + threadIdx.x] = 1;
            nulli[i + threadIdx.x] = 1;
            nulld[i + threadIdx.x] = 1;
        }
    }
    __syncthreads();

    mwavefronts[matr_idx] = 0;
    nullm[s] = 0;
    limitsm[s] = 0;
    limitsm[s + 1] = 0;

    while (true) {

        if (nullm[s]) {
            if (num_null_steps > max_score) {
                finish = 1; //done
            } else {
                finish = 0; // not done
            }
        } else {
            bool end_reached;
            int k;
            end_reached = false;

            for (k = limitsm[limits_idx]; k <= limitsm[limits_idx + 1]; ++k) {
                
                int16_t offset = mwavefronts[matr_idx + k];
               // printf("offset %d\n", offset);
                if (offset < 0) {continue;}

                int diffs = 0;
                int equal_chars = 0;
                unsigned int tid = threadIdx.x;
                
                diffs = count_chars(&pattern[offset - k], &text[offset], pattern_len);
                while((diffs == 0) && (offset-k+NUM_THREADS) <= pattern_len && (offset+NUM_THREADS) <= text_len){
                    offset += NUM_THREADS;
                    diffs = count_chars(&pattern[offset - k], &text[offset], pattern_len);
                }
                
                if(tid == 0){
                    if(!diffs) offset += NUM_THREADS;
                    if(offset-k < pattern_len) {
                        for(int i = offset; i < pattern_len; i++){
                            if(pattern[i-k] == text[i]){
                                equal_chars++;
                            }else break;
                        }
                    }
                }

                if(tid==0){
                    offset += equal_chars;
                    mwavefronts[matr_idx + k] = offset;
                }
            }

            // Select end component
            switch (component_end) {
                case matrix_M: {
                    // Check diagonal/offset
                    if (limitsm[limits_idx] > alignment_k || alignment_k > limitsm[limits_idx + 1]) {
                        end_reached = 0; // Not done
                    }else{
                         int16_t moffset;
                        moffset = mwavefronts[matr_idx + alignment_k];
                        if (moffset < alignment_offset) end_reached = 0; // Not done
                        else end_reached = 1;
                    }
                    break;
                }
                case matrix_I1: {
                    // Fetch I1-wavefront & check diagonal/offset
                    if (limitsi[limits_idx] > alignment_k || alignment_k > limitsi[limits_idx + 1]) end_reached = 0; // Not done
                    else{
                         int16_t i1offset;
                        i1offset = iwavefronts[matr_idx + alignment_k];
                        if (i1offset < alignment_offset) end_reached = 0; // Not done
                        else end_reached = 1;
                    }
                    break;
                }
                
                case matrix_D1: {
                    // Fetch D1-wavefront & check diagonal/offset
                    if (limitsd[limits_idx] > alignment_k || alignment_k > limitsd[limits_idx + 1]) end_reached = 0; // Not done
                    else{
                         int16_t d1offset;
                        d1offset = dwavefronts[matr_idx + alignment_k];
                        if (d1offset < alignment_offset) end_reached = 0; // Not done
                        else end_reached = 1;
                    }
                    break;
                }
                default:
                break;
            }
            if (end_reached) {
                finish = 1;
            }else finish = 0;
        }

        if (finish) {
            if(threadIdx.x == 0){
                score_d[blockIdx.x] = s;
                //printf("score %d - blockIdx.x %d\n", score_d[blockIdx.x], blockIdx.x);
                
                // Prepare cigar
                int end_offset = max_op - 1;
                int begin_offset = max_op - 2;
                operations[end_offset] = '\0';
                
                matrix_type component_begin = matrix_M;
                int k = alignment_k;
                int h = alignment_offset;
                int v = alignment_offset - alignment_k;
                int16_t offset = alignment_offset;
                // Account for ending insertions/deletions
                if (component_end == matrix_M) {
                    if (v < pattern_len) {
                        int i = pattern_len - v;
                        while (i > 0) {operations[begin_offset--] = 'D'; --i;};
                    }
                    if (h < text_len) {
                        int i = text_len - h;
                        while (i > 0) {operations[begin_offset--] = 'I'; --i;};                                                                              
                    }
                }
                
                // Trace the alignment back
                while (v > 0 && h > 0 && s > 0) {
                    short cigar_mismatch = (s - mismatch)*wf_length + wf_length/2;
                    short cigar_gapopen = (s - gap_opening - gap_extension)*wf_length + wf_length/2;
                    short cigar_gapextend = (s - gap_extension)*wf_length + wf_length/2;
                    
                    short cigar_limits_mism = (s - mismatch)*4;
                    short cigar_limits_gapopen = (s - gap_opening - gap_extension)*4;
                    short cigar_limits_gapextend = (s - gap_extension)*4;

                    short gap_open2 = s + 2;
                    short gap_extend2 = s + 1;
                    int64_t max_all;
                    
                    switch (component_begin) {
                        case matrix_M: {
                            
                            int64_t misms;
                            if ((s - mismatch) < 0) misms = OFFSET_NULL; 
                            else{
                                if (!nullm[s - mismatch] && limitsm[cigar_limits_mism] <= k && k <= limitsm[cigar_limits_mism + 1]) {
                                    misms = ((((int64_t)(mwavefronts[cigar_mismatch + k] + 1)) << 4) | bt_M);
                                } else {
                                    misms = OFFSET_NULL;
                                }
                            }

                            int64_t ins_open;
                            if ((s - gap_opening - gap_extension) < 0) ins_open = OFFSET_NULL;
                            else{
                                if (!nullm[s - gap_opening - gap_extension] && limitsm[cigar_limits_gapopen] <= k-1 && k-1 <= limitsm[cigar_limits_gapopen + 1]) {
                                    ins_open = ((((int64_t)(mwavefronts[cigar_gapopen + k - 1] + 1)) << 4) | bt_I1_open);
                                } else {
                                    ins_open = OFFSET_NULL;
                                }
                            }

                            int64_t ins_ext;
                            if ((s - gap_extension) < 0) ins_ext = OFFSET_NULL;
                            else{
                                if (!nulli[s - gap_extension] && limitsi[cigar_limits_gapextend] <= k-1 && k-1 <= limitsi[cigar_limits_gapextend + 1]) {
                                    ins_ext = ((((int64_t)(iwavefronts[cigar_gapextend + k - 1] + 1)) << 4) | bt_I1_ext);
                                } else {
                                    ins_ext = OFFSET_NULL;
                                }
                            }
                            
                            int64_t max_ins = MAX(ins_open,ins_ext);
                            int64_t del_open;
                            if ((s - gap_opening - gap_extension) < 0) del_open = OFFSET_NULL;
                            else{
                                if (!nullm[s - gap_opening - gap_extension] && limitsm[cigar_limits_gapopen] <= k+1 && k+1 <= limitsm[cigar_limits_gapopen + 1]) {
                                    del_open = ((((int64_t)(mwavefronts[cigar_gapopen + k + 1])) << 4) | bt_D1_open);
                                } else {
                                    del_open = OFFSET_NULL;
                                }
                            }

                            int64_t del_ext;
                            if ((s - gap_extension) < 0) del_ext = OFFSET_NULL;
                            else{
                                if (!nulld[s - gap_extension] && limitsd[cigar_limits_gapextend] <= k+1 && k+1 <= limitsd[cigar_limits_gapextend + 1]) {
                                    del_ext = ((((int64_t)(dwavefronts[cigar_gapextend + k + 1])) << 4) | bt_D1_ext);
                                } else {
                                    del_ext = OFFSET_NULL;
                                }
                            }
                            int64_t max_del = MAX(del_open, del_ext);
                            max_all = MAX(misms, MAX(max_ins, max_del));
                            break;
                        }

                        case matrix_I1: {
                            int64_t ins_open;
                            if ((s - gap_opening - gap_extension) < 0) ins_open = OFFSET_NULL;
                            if (!nullm[s - gap_opening - gap_extension] && limitsm[cigar_limits_gapopen] <= k-1 && k-1 <= limitsm[cigar_limits_gapopen + 1]) {
                                ins_open = ((((int64_t)(mwavefronts[cigar_gapopen + k - 1] + 1)) << 4) | bt_I1_open);
                            } else {
                                ins_open = OFFSET_NULL;
                            }

                            int64_t ins_ext;
                            if ((s - gap_extension) < 0) ins_ext = OFFSET_NULL;
                            if (!nulli[s - gap_extension] && limitsi[cigar_limits_gapextend] <= k-1 && k-1 <= limitsi[cigar_limits_gapextend + 1]) {
                                ins_ext = ((((int64_t)(iwavefronts[cigar_gapextend + k - 1] + 1)) << 4) | bt_I1_ext);
                            } else {
                                ins_ext = OFFSET_NULL;
                            }

                            max_all = MAX(ins_open, ins_ext);
                            break;
                        }

                        case matrix_D1: {
                            int64_t del_open;
                            if ((s - gap_opening - gap_extension) < 0) del_open = OFFSET_NULL;
                            if (!nullm[s - gap_opening - gap_extension] && limitsm[cigar_limits_gapopen] <= k+1 && k+1 <= limitsm[cigar_limits_gapopen + 1]) {
                                del_open = ((((int64_t)(mwavefronts[cigar_gapopen + k + 1])) << 4) | bt_D1_open);
                            } else {
                                del_open = OFFSET_NULL;
                            }

                            int64_t del_ext;
                            if ((s - gap_extension) < 0) del_ext = OFFSET_NULL;
                            if (!nulld[s - gap_extension] && limitsd[cigar_limits_gapextend] <= k+1 && k+1 <= limitsd[cigar_limits_gapextend + 1]) {
                                del_ext = ((((int64_t)(dwavefronts[cigar_gapextend + k + 1])) << 4) | bt_D1_ext);
                            } else {
                                del_ext = OFFSET_NULL;
                            }

                            max_all = MAX(del_open, del_ext);
                            break;
                        }
                        
                        default:
                            break;
                    }

                    //wavefront_bt_Matches(wf_aligner,k,offset,num_matches,cigar);
                    if (max_all < 0) break; // No source
                    // Traceback Matches
                    if (component_begin == matrix_M) {
                        int max_offset = ((max_all) >> 4);
                        int num_matches = offset - max_offset;
                        // Update offset first
                        begin_offset -= num_matches;
                        while (num_matches >= 1) {
                            operations[begin_offset + 1] = 'M';
                            num_matches--;
                        }

                        // Remaining matches
                        for (int i = 0; i < num_matches; i++) {
                            operations[i] = 'M';
                        }

                        offset = max_offset;
                        // Update coordinates
                        v = offset - k;
                        h = offset;
                        if (v <= 0 || h <= 0) break;
                    }
                    // Traceback Operation

                    bt_type_t backtrace_type = (bt_type_t)((max_all) & 0x000000000000000Fl);

                    switch (backtrace_type) {
                    case bt_M:
                        s = cigar_mismatch;
                        component_begin = matrix_M;
                        break;
                    case bt_I1_open:
                        s = cigar_gapopen;
                        component_begin = matrix_M;
                        break;
                    case bt_I1_ext:
                        s = cigar_gapextend;
                        component_begin = matrix_I1;
                        break;
                    case bt_I2_open:
                        s = gap_open2;
                        component_begin = matrix_M;
                        break;
                    case bt_I2_ext:
                        s = gap_extend2;
                        component_begin = matrix_I2;
                        break;
                    case bt_D1_open:
                        s = cigar_gapopen;
                        component_begin = matrix_M;
                        break;
                    case bt_D1_ext:
                        s = cigar_gapextend;
                        component_begin = matrix_D1;
                        break;
                    case bt_D2_open:
                        s = gap_open2;
                        component_begin = matrix_M;
                        break;
                    case bt_D2_ext:
                        s = gap_extend2;
                        component_begin = matrix_D2;
                        break;
                    default:
                        break;
                    }


                    switch (backtrace_type) {
                        case bt_M:
                            operations[begin_offset--] = 'X';
                            --offset;
                            break;
                        case bt_I1_open:
                        case bt_I1_ext:
                        case bt_I2_open:
                        case bt_I2_ext:
                            operations[begin_offset--] = 'I';
                            --k; 
                            --offset;
                            break;
                        case bt_D1_open:
                        case bt_D1_ext:
                        case bt_D2_open:
                        case bt_D2_ext:
                            operations[begin_offset--] = 'D';
                            ++k;
                            break;
                        default:
                            break;
                        }
                    // Update coordinates
                    v = offset-k;
                    h = offset;
                }
 
                // Account for last operations
                if (component_begin == matrix_M) {
                    if (v > 0 && h > 0) {
                        // Account for beginning series of matches
                        int num_matches = MIN(v,h);
                        // Update offset first
                        begin_offset -= num_matches;
                        while (num_matches >= 1) {
                            operations[begin_offset + 1] = 'M';
                            num_matches--;
                        }
                        // Remaining matches
                        for (int i = 0; i < num_matches; i++) {
                            operations[i] = 'M';
                        }        
                        v -= num_matches;
                        h -= num_matches;
                    }
                    // Account for beginning insertions/deletions
                    while (v > 0) {operations[begin_offset--] = 'D'; --v;};
                    while (h > 0) {operations[begin_offset--] = 'I'; --h;};
                } 

                // Set CIGAR
                begin_offset++;
                printf("%s \n", operations);
                break;
            }
        }

        s++;
        matr_idx = s*wf_length + wf_length/2;
        limits_idx = s*4;
        
        //wavefront compute affine
        int gap_open = (s - gap_opening - gap_extension)*wf_length + wf_length/2;
        int mism = (s - mismatch)*wf_length + wf_length/2;
        int gap_ext = (s - gap_extension)*wf_length + wf_length/2;

        int limits_gap_open = (s - gap_opening - gap_extension)*4;
        int limits_mism = (s - mismatch)*4;
        int limits_gap_ext = (s - gap_extension)*4;

        bool nullmism = 0;
        bool nullopen = 0;
        bool nulliext = 0;
        bool nulldext = 0;

        short lo_mism;
        short hi_mism;
        short lo_open;
        short hi_open;
        short lo_iext;
        short hi_iext;
        short lo_dext;
        short hi_dext;

        if((s - mismatch) < 0 || &mwavefronts[mism] == NULL){
            nullmism = 1;
            lo_mism = 1;
            hi_mism = -1;
        }else if(nullm[s - mismatch]){
            nullmism = 1;
            lo_mism = 1;
            hi_mism = -1;
        }else{
            nullmism = nullm[s-mismatch];
            lo_mism = limitsm[limits_mism];
            hi_mism = limitsm[limits_mism + 1];
        }

        if((s - gap_opening - gap_extension) < 0 || &mwavefronts[gap_open] == NULL){
            nullopen = 1;
            lo_open = 1;
            hi_open = -1;
        }else if(nullm[s - gap_opening - gap_extension]){
            nullopen = 1;
            lo_open = 1;
            hi_open = -1;
        }else{
            nullopen = nullm[s-gap_opening-gap_extension];
            lo_open = limitsm[limits_gap_open];
            hi_open = limitsm[limits_gap_open + 1];
        }
        
        if((s - gap_extension) < 0 || &iwavefronts[gap_ext] == NULL){
            nulliext = 1;
            lo_iext = 1;
            hi_iext = -1;
        }else if(nulli[s - gap_extension]){
            nulliext = 1;
            lo_iext = 1;
            hi_iext = -1;
        }else{
            nulliext = nulli[s-gap_extension];
            lo_iext = limitsi[limits_gap_ext];
            hi_iext = limitsi[limits_gap_ext + 1];
        }

        if((s - gap_extension) < 0 || &dwavefronts[gap_ext] == NULL){
            nulldext = 1;
            lo_dext = 1;
            hi_dext = -1;
        }else if(nulld[s - gap_extension]){
            nulldext = 1;
            lo_dext = 1;
            hi_dext = -1;
        }else{
            nulldext = nulld[s - gap_extension];
            lo_dext = limitsd[limits_gap_ext];
            hi_dext = limitsd[limits_gap_ext + 1];
        }

        if (nullmism && nullopen && nulliext && nulldext) {
            num_null_steps++; // Increment null-steps
            // Nullify Wavefronts
            nullm[s] = 1;
            nulli[s] = 1;
            nulld[s] = 1;
        }else{
            
            num_null_steps = 0;
            int hi, lo;
            int min_lo;
            int max_hi;
            min_lo = lo_mism;
            max_hi = hi_mism;
            
            if (min_lo > lo_open-1 && !nullopen) min_lo = lo_open-1;
            if (max_hi < hi_open+1 && !nullopen) max_hi = hi_open+1;
            if (min_lo > lo_iext+1 && !nulliext) min_lo = lo_iext+1;
            if (max_hi < hi_iext+1 && !nulliext) max_hi = hi_iext+1;
            if (min_lo > lo_dext-1 && !nulldext) min_lo = lo_dext-1;
            if (max_hi < hi_dext-1 && !nulldext) max_hi = hi_dext-1;
            lo = min_lo;
            hi = max_hi;

            int effective_lo = lo;
            int effective_hi = hi;
            int eff_lo = effective_lo - (max_score + 1);
            int eff_hi = effective_hi + (max_score + 1);
            effective_lo = MIN(eff_lo, historic_min_lo);
            effective_hi = MAX(eff_hi, historic_max_hi);
            
            // Allocate M-Wavefront
            historic_min_lo = effective_lo;
            historic_max_hi = effective_hi;
            
            limitsm[limits_idx] = lo;
            limitsm[limits_idx + 1] = hi;

            // Allocate I1-Wavefront            
            if (!nullopen || !nulliext) {
                limitsi[limits_idx] = lo;
                limitsi[limits_idx + 1] = hi;
            } else {
                nulli[s] = 1;
            }
            // Allocate D1-Wavefront
            if (!nullopen || !nulldext) {
                limitsd[limits_idx] = lo;
                limitsd[limits_idx + 1] = hi;
            } else {
                nulld[s] = 1;
            }
            
            if(threadIdx.x == 0){
                if (!nullmism) {                
                    if (limitsm[limits_mism + 3] >= hi){ 
                    }else{
                        // Initialize lower elements
                        int max_init = MAX(limitsm[limits_mism + 3], limitsm[limits_mism + 1]);
                        int k;

                        for (k = max_init + 1; k <= hi; k += blockDim.x) {
                            if(k + threadIdx.x <= hi){
                                mwavefronts[mism + k + threadIdx.x] = OFFSET_NULL;
                            }
                        }
                        
                        // Set new maximum
                        limitsm[limits_mism + 3] = hi;
                    }   

                    //------------------------------------------------------------------------------------//
                    
                    if (limitsm[limits_mism + 2] <= lo){
                    }else{  
                        // Initialize lower elements
                        int min_init = MIN(limitsm[limits_mism + 2], limitsm[limits_mism]);
                        int k;

                        for (k = lo; k < min_init; k += blockDim.x) {
                            if(k + threadIdx.x < min_init){
                                mwavefronts[mism + k + threadIdx.x] = OFFSET_NULL;
                            }
                        }
                        // Set new minimum
                        limitsm[limits_mism + 2] = lo;

                    }   
                }
                if (!nullopen) {
                    if (limitsm[limits_gap_open + 3] >= hi+1){
                    }else{
                        // Initialize lower elements
                        int max_init = MAX(limitsm[limits_gap_open + 3], limitsm[limits_gap_open + 1]);
                        int k;

                        for (k = max_init + 1; k <= hi + 1; k += blockDim.x) {
                            if(k + threadIdx.x <= hi){
                                mwavefronts[gap_open + k + threadIdx.x] = OFFSET_NULL;                    
                            }
                        }
                        // Set new maximum
                        limitsm[limits_gap_open + 3] = hi + 1;
                    }
                    
                    //------------------------------------------------------------------------------------//

                    if (limitsm[limits_gap_open + 2] <= lo - 1){
                    }else{
                        // Initialize lower elements
                        int min_init = MIN(limitsm[limits_gap_open + 2], limitsm[limits_gap_open]);
                        int k;

                        for (k = lo - 1;k < min_init; k += blockDim.x) {
                            if(k + threadIdx.x < min_init){
                                mwavefronts[gap_open + k + threadIdx.x] = OFFSET_NULL;
                            }
                        }
                        // Set new minimum
                        limitsm[limits_gap_open + 2] = lo - 1;
                    }
                }
                
                if (!nulliext) {
                    
                    if (limitsi[limits_gap_ext + 3] >= hi){
                    }else{
                        // Initialize lower elements
                        int max_init = MAX(limitsi[limits_gap_ext + 3], limitsi[limits_gap_ext + 1]);
                        int k;

                        for (k = max_init + 1; k <= hi; k += blockDim.x) {
                            if(k + threadIdx.x <= hi){
                                iwavefronts[gap_ext + k + threadIdx.x] = OFFSET_NULL;
                            }
                        }
                        // Set new maximum
                        limitsi[limits_gap_ext + 3] = hi;
                    }
                    
                    //------------------------------------------------------------------------------------//
                    if (limitsi[limits_gap_ext + 2] <= lo - 1){
                    }else{
                        // Initialize lower elements
                        int min_init = MIN(limitsi[limits_gap_ext + 2], limitsi[limits_gap_ext]);
                        int k;

                        for (k = lo - 1; k < min_init; k += blockDim.x) {
                            if(k + threadIdx.x < min_init){
                                iwavefronts[gap_ext + k + threadIdx.x] = OFFSET_NULL;
                            }
                        }
                        // Set new minimum
                        limitsi[limits_gap_ext + 2] = lo - 1;
                    }       
                }
                
                if (!nulldext) {
                    if (limitsd[limits_gap_ext + 3] >= hi+1){
                    }else{
                        // Initialize lower elements
                        int max_init = MAX(limitsd[limits_gap_ext + 3], limitsd[limits_gap_ext + 1]);
                        int k;

                        for (k = max_init + 1; k <= hi + 1; k += blockDim.x) {
                            if(k + threadIdx.x <= hi + 1){
                                dwavefronts[gap_ext + k + threadIdx.x] = OFFSET_NULL;
                            }
                        }
                        // Set new maximum
                        limitsd[limits_gap_ext + 3] = hi + 1;
                    }

                    //------------------------------------------------------------------------------------//
                    if (limitsd[limits_gap_ext + 2] <= lo){
                    }else{
                        // Initialize lower elements
                        int min_init = MIN(limitsd[limits_gap_ext + 2], limitsi[limits_gap_ext]);
                        int k;
                        
                        for (k = lo; k < min_init; k += blockDim.x) {
                            if(k + threadIdx.x < min_init){
                                dwavefronts[gap_ext + k + threadIdx.x] = OFFSET_NULL;
                            }
                        }
                        // Set new minimum
                        limitsd[limits_gap_ext + 2] = lo;
                    }
                }
            }
            
            __syncthreads();

            // Compute-Next kernel loop
            int tidx = threadIdx.x;
            for(int i = 0; i <= (hi-lo); i += blockDim.x){
                int idx = tidx + i;
                if(idx <= (hi-lo)){
                    // Update I
                    short ins_o = (s-gap_opening-gap_extension < 0) ? OFFSET_NULL : mwavefronts[gap_open + idx + lo - 1];
                    short ins_e = (s-gap_extension < 0) ? OFFSET_NULL : iwavefronts[gap_ext + idx + lo - 1];
                    short ins = MAX(ins_o, ins_e) + 1;
                    iwavefronts[matr_idx + idx + lo] = ins;

                    
                    // Update D
                    short del_o = (s-gap_opening-gap_extension < 0) ? OFFSET_NULL : mwavefronts[gap_open + idx + lo + 1];
                    short del_e = (s-gap_extension < 0) ? OFFSET_NULL : dwavefronts[gap_ext + idx + lo + 1];
                    short del = MAX(del_o, del_e);
                    dwavefronts[matr_idx + idx + lo] = del;
                    
                    // Update M
                    short mism_m = (s-mismatch < 0) ? OFFSET_NULL : mwavefronts[mism + idx + lo];
                    short max = MAX(del, MAX(mism_m + 1, ins));
                    ushort h = max; 
                    ushort v = max - idx + lo; 
                    if (h > text_len) max = OFFSET_NULL;
                    if (v > pattern_len) max = OFFSET_NULL;
                    mwavefronts[matr_idx + idx + lo] = max;
                }
            }

            //wavefront_compute_process_ends(wf_aligner,&wavefront_set,score);
            // Trim ends from non-null WFs
            if (&mwavefronts[matr_idx]){
                int k;
                int lo = limitsm[limits_idx];
                for (k = limitsm[limits_idx + 1]; k >= lo; --k) {
                    // Fetch offset
                    int16_t offset = mwavefronts[matr_idx + k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                limitsm[limits_idx + 1] = k; // Set new hi
                limitsm[limits_idx + 3] = k;
                // Trim from lo
                 int hi;
                hi = limitsm[limits_idx + 1];
                for (k = limitsm[limits_idx]; k <= hi; ++k) {
                    // Fetch offset
                    int16_t offset = mwavefronts[matr_idx + k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset - k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                limitsm[limits_idx] = k; // Set new lo
                limitsm[limits_idx + 2] = k;
                nullm[s] = (limitsm[limits_idx] > limitsm[limits_idx + 1]);
            }

            if (&iwavefronts[matr_idx]){
                int k;
                int lo = limitsi[limits_idx];
                for (k = limitsi[limits_idx + 1]; k >= lo; --k) {
                    // Fetch offset
                    int16_t offset = iwavefronts[matr_idx + k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset - k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                limitsi[limits_idx + 1] = k; // Set new hi
                limitsi[limits_idx + 3] = k;
                // Trim from lo
                int hi = limitsi[limits_idx + 1];
                for (k = limitsi[limits_idx]; k <= hi; ++k) {
                    // Fetch offset
                    int16_t offset = iwavefronts[matr_idx + k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset - k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                limitsi[limits_idx] = k; // Set new lo
                limitsi[limits_idx+2] = k;
                nulli[s] = (limitsi[limits_idx] > limitsi[limits_idx + 1]);
            }

            if (&dwavefronts[matr_idx]){
                int k;
                int lo = limitsd[limits_idx];
                for (k = limitsd[limits_idx + 1]; k >= lo; --k) {
                    // Fetch offset
                    int16_t offset = dwavefronts[matr_idx + k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset - k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                limitsd[limits_idx + 1] = k; // Set new hi
                limitsd[limits_idx + 3] = k;
                // Trim from lo
                int hi = limitsd[limits_idx + 1];
                for (k = limitsd[limits_idx]; k <= hi; ++k) {
                    // Fetch offset
                    int16_t offset = dwavefronts[matr_idx + k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative 
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                limitsd[limits_idx] = k; // Set new lo
                limitsd[limits_idx + 2] = k;
                nulld[s] = (limitsd[limits_idx] > limitsd[limits_idx + 1]);
            }  
        }
    }
}

__device__ int count_chars(char *pattern, char *text, int seq_len){
    

    __shared__ int sdata[NUM_THREADS];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    int equal_chars = 0;

    sdata[tid] = 0;
    __syncthreads();

    if(pattern[tid] != text[tid]){
        sdata[tid] = 1;
    }
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
    __syncthreads();
    }

    if (tid == 0){
        equal_chars = sdata[0];
        return equal_chars;
    } 
}
