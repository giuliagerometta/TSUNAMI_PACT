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

__global__ void wfa_dispatcher(char *pattern_d, char* text_d, short *score_d, short *mwavefronts_d, 
                                short *iwavefronts_d, short* dwavefronts_d, int max_op, int num_couples, 
                                bool *nullm_d, bool *nulli_d, bool *nulld_d, short *limitsm_d, short *limitsi_d, short *limitsd_d,
                                int pattern_len, int text_len, int num_wavefronts, int wf_length, int max_score, short mismatch, short gap_opening, short gap_extension, int hi, int lo);

__device__ int count_chars(char *pattern, char *text, int seq_len);

int main(int argc, char const *argv[]){

    if(argc!=6){
		printf("ERROR! Please specify in order: mismatch, gap_opening, gap_extension, file name, check\n");
		return 0;
	}

    int num_couples, seq_len, error_rate;
    bool check;
    char *pattern, *text, *pattern_d, *text_d;
    short *mwavefronts_d, *iwavefronts_d, *dwavefronts_d;
    bool *nullm_d, *nulli_d, *nulld_d;
    short *limitsm_d, *limitsi_d, *limitsd_d;
    short *score_d, *score;

	short mismatch = atoi(argv[1]);
	short gap_opening = atoi(argv[2]);
	short gap_extension = atoi(argv[3]);
    FILE* fp = fopen(argv[4], "r");
    check = atoi(argv[5]);

    if(fp==NULL){
		printf("ERROR: Cannot open file.\n");
		return 1;
	}

    fscanf(fp, "%d %d %d", &num_couples, &seq_len, &error_rate);
    int pattern_len = seq_len;
    int text_len = seq_len;

    score = (short*)malloc(sizeof(short)*num_couples);
    pattern = (char*)malloc(sizeof(char)*pattern_len*num_couples);
    text = (char*)malloc(sizeof(char)*text_len*num_couples);
    for(int i = 0; i<num_couples; i++){
		fscanf(fp, "%s", pattern+(i*pattern_len));
		fscanf(fp, "%s", text+(i*text_len));
	}

    int max_score = MAX(gap_opening + gap_extension, mismatch) + 1;
    int num_wavefronts = max_score; //max_score_misms + max_score_indel + 1;
    int max_op = 2 * (pattern_len + text_len);
    int hi = 0;
    int lo = 0;
    int eff_lo = lo - (max_score + 1);
    int eff_hi = hi + (max_score + 1);
    lo = MIN(eff_lo, 0);
    hi = MAX(eff_hi, 0);
    int wf_length = hi - lo + 1;
    wf_length*=error_rate;

    
    CHECK(cudaDeviceReset());
    CHECK(cudaMalloc(&pattern_d, sizeof(char)*pattern_len*num_couples));
    CHECK(cudaMalloc(&text_d, sizeof(char)*text_len*num_couples));
    CHECK(cudaMalloc(&score_d, sizeof(short)*num_couples));
    CHECK(cudaMalloc(&mwavefronts_d, sizeof(short)*num_wavefronts*wf_length*num_couples));
    CHECK(cudaMalloc(&dwavefronts_d, sizeof(short)*num_wavefronts*wf_length*num_couples));
    CHECK(cudaMalloc(&iwavefronts_d, sizeof(short)*num_wavefronts*wf_length*num_couples));
    CHECK(cudaMalloc(&nullm_d, sizeof(bool)*num_wavefronts*num_couples));
    CHECK(cudaMalloc(&nulli_d, sizeof(bool)*num_wavefronts*num_couples));
    CHECK(cudaMalloc(&nulld_d, sizeof(bool)*num_wavefronts*num_couples));
    CHECK(cudaMalloc(&limitsm_d, sizeof(short)*num_wavefronts*4*num_couples));
    CHECK(cudaMalloc(&limitsi_d, sizeof(short)*num_wavefronts*4*num_couples));
    CHECK(cudaMalloc(&limitsd_d, sizeof(short)*num_wavefronts*4*num_couples));

    CHECK(cudaMemcpy(pattern_d, pattern, sizeof(char)*num_couples*pattern_len, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(text_d, text, sizeof(char)*num_couples*text_len, cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(num_couples, 1, 1);
	dim3 threadsPerBlock(NUM_THREADS, 1, 1);

    std::chrono::high_resolution_clock::time_point start = NOW;

    wfa_dispatcher<<<blocksPerGrid, threadsPerBlock>>>(pattern_d, text_d, score_d, mwavefronts_d, iwavefronts_d, 
                    dwavefronts_d, max_op, num_couples, nullm_d, nulli_d, nulld_d, limitsm_d, limitsi_d, limitsd_d,
                    pattern_len, text_len, num_wavefronts, wf_length, max_score, mismatch, gap_opening, gap_extension, hi, lo);
    
    CHECK_KERNELCALL();
    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point end = NOW;
    std::chrono::duration<double> time_temp = (end - start);
    
    CHECK(cudaMemcpy(score, score_d, sizeof(short)*num_couples, cudaMemcpyDeviceToHost));

    long double gcups = (pattern_len*text_len);
    gcups/=(1E9);
    gcups/=(time_temp.count());
    gcups*=num_couples;
    printf("Time: %lf\n", time_temp.count());
    printf("Estimated GCUPS gpu: : %Lf\n", gcups);

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
        short *smarco_score = (short*)malloc(sizeof(short)*num_couples);
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
        long double gcups_cpu = (pattern_len*text_len);
        gcups_cpu/=(1E9);
        gcups_cpu/=(time_temp_cpu.count());
        gcups_cpu*=num_couples;
        printf("Estimated GCUPS sw: %Lf\n", gcups_cpu);

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


__global__ void wfa_dispatcher(char *pattern_d, char* text_d, short *score_d, short *mwavefronts_d, 
                                short *iwavefronts_d, short* dwavefronts_d, int max_op, int num_couples, 
                                bool *nullm_d, bool *nulli_d, bool *nulld_d, short *limitsm_d, short *limitsi_d, short *limitsd_d,
                                int pattern_len, int text_len, int num_wavefronts, int wf_length, int max_score, short mismatch, short gap_opening, short gap_extension, int hi, int lo){

    char *pattern = &pattern_d[pattern_len*blockIdx.x];
    char *text = &text_d[text_len*blockIdx.x];
    short *mwavefronts = &mwavefronts_d[num_wavefronts*wf_length*blockIdx.x];
    short *iwavefronts = &iwavefronts_d[num_wavefronts*wf_length*blockIdx.x];
    short *dwavefronts = &dwavefronts_d[num_wavefronts*wf_length*blockIdx.x];
    bool *nullm = &nullm_d[num_wavefronts*blockIdx.x]; 
    bool *nulli = &nulli_d[num_wavefronts*blockIdx.x]; 
    bool *nulld = &nulld_d[num_wavefronts*blockIdx.x];
    short *limitsm = &limitsm_d[4*num_wavefronts*blockIdx.x]; //lo = 0, hi = 1, wf_elements_init_min = 2,  wf_elements_init_max = 3
    short *limitsi = &limitsi_d[4*num_wavefronts*blockIdx.x];
    short *limitsd = &limitsd_d[4*num_wavefronts*blockIdx.x];

    int s = 0;

    short num_null_steps = 0;
    short historic_max_hi = 0;
    short historic_min_lo = 0;
    bool finish = false;
    short alignment_k = text_len - pattern_len;
    short alignment_offset = text_len;
    matrix_type component_end = matrix_M;
    short matr_idx = (s%num_wavefronts)*wf_length + wf_length/2;
    short limits_idx = (s% num_wavefronts)*4;

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
        
        if (nullm[s%num_wavefronts]) {
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
                
                short offset = mwavefronts[matr_idx + k];
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
                        short moffset;
                        moffset = mwavefronts[matr_idx + alignment_k];
                        if (moffset < alignment_offset) {end_reached = 0;} // Not done
                        else end_reached = 1;
                    }
                    break;
                }
                case matrix_I1: {
                    // Fetch I1-wavefront & check diagonal/offset
                    if (limitsi[limits_idx] > alignment_k || alignment_k > limitsi[limits_idx + 1]) end_reached = 0; // Not done
                    else{
                         short i1offset;
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
                         short d1offset;
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
            }
            break;
        }

        s++;
        matr_idx = (s%num_wavefronts)*wf_length + wf_length/2;
        limits_idx = (s% num_wavefronts)*4;
        nullm[s%num_wavefronts] = 0;
        nulli[s%num_wavefronts] = 0;
        nulld[s%num_wavefronts] = 0;
        //wavefront compute affine
        int gap_open = ((s - gap_opening - gap_extension)%num_wavefronts)*wf_length + wf_length/2;
        int mism = ((s - mismatch)%num_wavefronts)*wf_length + wf_length/2;
        int gap_ext = ((s - gap_extension)%num_wavefronts)*wf_length + wf_length/2;

        int limits_gap_open = ((s - gap_opening - gap_extension)%num_wavefronts)*4;
        int limits_mism = ((s - mismatch)%num_wavefronts)*4;
        int limits_gap_ext = ((s - gap_extension)%num_wavefronts)*4;

        bool nullmism = 0;
        bool nullopen = 0;
        bool nulliext = 0;
        bool nulldext = 0;

        short lo_mism, hi_mism, lo_open, hi_open, lo_iext, hi_iext, lo_dext, hi_dext;

        if((s - mismatch) < 0 || &mwavefronts[mism] == NULL){
            nullmism = 1;
            lo_mism = 1;
            hi_mism = -1;
        }else if(nullm[(s - mismatch)%num_wavefronts]){
            nullmism = 1;
            lo_mism = 1;
            hi_mism = -1;
        }else{
            nullmism = nullm[(s-mismatch)%num_wavefronts];
            lo_mism = limitsm[limits_mism];
            hi_mism = limitsm[limits_mism + 1];
        }

        if((s - gap_opening - gap_extension) < 0 || &mwavefronts[gap_open] == NULL){
            nullopen = 1;
            lo_open = 1;
            hi_open = -1;
        }else if(nullm[(s - gap_opening - gap_extension)%num_wavefronts]){
            nullopen = 1;
            lo_open = 1;
            hi_open = -1;
        }else{
            nullopen = nullm[(s-gap_opening-gap_extension)%num_wavefronts];
            lo_open = limitsm[limits_gap_open];
            hi_open = limitsm[limits_gap_open + 1];
        }
        
        if((s - gap_extension) < 0 || &iwavefronts[gap_ext] == NULL){
            nulliext = 1;
            lo_iext = 1;
            hi_iext = -1;
        }else if(nulli[(s - gap_extension)%num_wavefronts]){
            nulliext = 1;
            lo_iext = 1;
            hi_iext = -1;
        }else{
            nulliext = nulli[(s-gap_extension)%num_wavefronts];
            lo_iext = limitsi[limits_gap_ext];
            hi_iext = limitsi[limits_gap_ext + 1];
        }

        if((s - gap_extension) < 0 || &dwavefronts[gap_ext] == NULL){
            nulldext = 1;
            lo_dext = 1;
            hi_dext = -1;
        }else if(nulld[(s - gap_extension)%num_wavefronts]){
            nulldext = 1;
            lo_dext = 1;
            hi_dext = -1;
        }else{
            nulldext = nulld[(s-gap_extension)%num_wavefronts];
            lo_dext = limitsd[limits_gap_ext];
            hi_dext = limitsd[limits_gap_ext + 1];
        }
        if (nullmism && nullopen && nulliext && nulldext) {
            num_null_steps++; // Increment null-steps
            // Nullify Wavefronts
            nullm[s%num_wavefronts] = 1;
            nulli[s%num_wavefronts] = 1;
            nulld[s%num_wavefronts] = 1;
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
                nulli[s%num_wavefronts] = 1;
            }
            // Allocate D1-Wavefront
            if (!nullopen || !nulldext) {
                limitsd[limits_idx] = lo;
                limitsd[limits_idx + 1] = hi;
            } else {
                nulld[s%num_wavefronts] = 1;
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
                    iwavefronts[matr_idx + idx + lo] = MAX(ins_o, ins_e) + 1;
                    
                    // Update D
                    short del_o = (s-gap_opening-gap_extension < 0) ? OFFSET_NULL : mwavefronts[gap_open + idx + lo + 1];
                    short del_e = (s-gap_extension < 0) ? OFFSET_NULL : dwavefronts[gap_ext + idx + lo + 1];
                    dwavefronts[matr_idx + idx + lo] = MAX(del_o, del_e);
                    // Update M
            
                    mwavefronts[matr_idx + idx + lo] = MAX(dwavefronts[matr_idx + idx + lo], MAX(mwavefronts[mism + idx + lo] + 1, iwavefronts[matr_idx + idx + lo]));
                    
                    if (mwavefronts[matr_idx + idx + lo] > text_len) {mwavefronts[matr_idx + idx + lo] = OFFSET_NULL;}
                    if (mwavefronts[matr_idx + idx + lo] - idx + lo > pattern_len) {mwavefronts[matr_idx + idx + lo] = OFFSET_NULL;}
                }
            }

            //wavefront_compute_process_ends(wf_aligner,&wavefront_set,score);
            // Trim ends from non-null WFs
            
            if (&mwavefronts[matr_idx]){
                int k;
                int lo = limitsm[limits_idx];
                for (k = limitsm[limits_idx + 1]; k >= lo; --k) {
                    // Fetch offset
                    short offset = mwavefronts[matr_idx + k];
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
                    short offset = mwavefronts[matr_idx + k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset - k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                limitsm[limits_idx] = k; // Set new lo
                limitsm[limits_idx + 2] = k;
                nullm[s%num_wavefronts] = (limitsm[limits_idx] > limitsm[limits_idx + 1]);
            }

            if (&iwavefronts[matr_idx]){
                int k;
                int lo = limitsi[limits_idx];
                for (k = limitsi[limits_idx + 1]; k >= lo; --k) {
                    // Fetch offset
                    short offset = iwavefronts[matr_idx + k];
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
                    short offset = iwavefronts[matr_idx + k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset - k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                limitsi[limits_idx] = k; // Set new lo
                limitsi[limits_idx+2] = k;
                nulli[s%num_wavefronts] = (limitsi[limits_idx] > limitsi[limits_idx + 1]);
            }

            if (&dwavefronts[matr_idx]){
                int k;
                int lo = limitsd[limits_idx];
                for (k = limitsd[limits_idx + 1]; k >= lo; --k) {
                    // Fetch offset
                    short offset = dwavefronts[matr_idx + k];
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
                    short offset = dwavefronts[matr_idx + k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative 
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                limitsd[limits_idx] = k; // Set new lo
                limitsd[limits_idx + 2] = k;
                nulld[s%num_wavefronts] = (limitsd[limits_idx] > limitsd[limits_idx + 1]);
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
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
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
