#include <cstdlib>
#include <string>
#include <vector>
#include <cstring>
#include "common.h"


int main(int argc, char const *argv[]){

    double cpu_start, cpu_end, cpu_exectime;
    double gpu_start, gpu_end, gpu_exectime;
    int num_couples;
    penalties_t penalties;

    if(argc!=5){
		printf("Please specify in order: mismatch, gap_opening, gap_extension, file name\n");
		return 0;
	}

    penalties.mismatch = atoi(argv[1]);
	penalties.gap_open = atoi(argv[2]);
	penalties.gap_ext = atoi(argv[3]);
    penalties.match = 0; 

    FILE* fp = fopen(argv[4], "r");

    if(fp==NULL){
		printf("ERROR: Cannot open file.\n");
		return 1;
	}

    int seq_len;
    fscanf(fp, "%d %d", &num_couples, &seq_len);
    int pattern_len = seq_len;
    int text_len = seq_len;

    wf_t_cpu alignment; 
    wf_components_t_cpu wf;

    alignment.pattern = (char*)malloc(sizeof(char)*pattern_len);
    alignment.text = (char*)malloc(sizeof(char)*text_len);
    for(int i = 0; i<num_couples; i++){
		fscanf(fp, "%s", alignment.pattern+(i*pattern_len));
		fscanf(fp, "%s", alignment.text+(i*text_len));
	}
    alignment.score = 0;
    alignment.wf_elements_init_max = 0;
    alignment.wf_elements_init_min = 0;

    int max_score_scope = MAX(penalties.gap_open+penalties.gap_ext, penalties.mismatch) + 1;
    int num_wavefronts = max_score_scope;
    int max_op = 2*(pattern_len+text_len);
    matrix_type component_end = matrix_M;
    matrix_type component_begin = matrix_M;

    int hi = 0;
    int lo = 0;
    int eff_lo = lo - (max_score_scope + 1);
    int eff_hi = hi + (max_score_scope + 1);
    lo = MIN(eff_lo,0);
    hi = MAX(eff_hi,0);
    int wf_length = hi - lo + 1;

    wf.mwavefronts = (wf_t_cpu*)malloc(sizeof(wf_t_cpu)*num_wavefronts);
    wf.dwavefronts = (wf_t_cpu*)malloc(sizeof(wf_t_cpu)*num_wavefronts);
    wf.iwavefronts = (wf_t_cpu*)malloc(sizeof(wf_t_cpu)*num_wavefronts);
    wf.mwavefronts[0].offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
    wf.iwavefronts[0].offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
    wf.dwavefronts[0].offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
    wf.wavefront_null.offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
    wf.mwavefronts[0].offsets[0] = 0;
    wf.dwavefronts[0].null = 1;
    wf.iwavefronts[0].null = 1;
    wf.wavefront_null.hi = -1;
    wf.wavefront_null.lo = 1;
    wf.wavefront_null.null = 1;
    alignment.num_null_steps = 0;
    alignment.historic_max_hi = 0;
    alignment.historic_min_lo = 0;
    int score = alignment.score;
    wf.mwavefronts[score].lo = 0;
    wf.mwavefronts[score].hi = 0;
    bool finish = 0;
    int alignment_k = text_len-pattern_len;
    int16_t alignment_offset = text_len;
    int score_mod = score%max_score_scope;


    alignment.cigar.operations = (char*)malloc(sizeof(char)*max_op);
    alignment.cigar.max_operations = 2*(pattern_len+text_len);
    alignment.cigar.begin_offset = 0;
    alignment.cigar.end_offset = 0;
    alignment.cigar.score = INT16_MIN;

    wf.mwavefronts[0].bt_pcigar = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
    wf.iwavefronts[0].bt_pcigar = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
    wf.dwavefronts[0].bt_pcigar = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
    wf.mwavefronts[0].bt_prev = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
    wf.iwavefronts[0].bt_prev = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
    wf.dwavefronts[0].bt_prev = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
    wf.wavefront_null.bt_pcigar = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
    wf.wavefront_null.bt_prev = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);

    for(int i=-(wf_length/2); i<(wf_length/2); i++){
        wf.wavefront_null.offsets[i] = OFFSET_NULL;
        wf.wavefront_null.bt_pcigar[i] = 0;
        wf.wavefront_null.bt_prev[i] = 0;
    }

    while (true) {
        // Exact extend s-wavefront
        // Fetch m-wavefront
        
        if (wf.mwavefronts[score_mod].offsets == NULL) {
            if(alignment.num_null_steps > max_score_scope){
                finish = 1; //done
            }else{
                finish = 0; // not done
            }

        }else{
            bool end_reached = false;
            int k;
            
            for (k=wf.mwavefronts[score_mod].lo;k<=wf.mwavefronts[score_mod].hi;++k) {
                
                int16_t offset = wf.mwavefronts[score_mod].offsets[k]; //offset + k
                if (offset == OFFSET_NULL) {continue;}
                uint64_t* pattern_blocks = (uint64_t*)(alignment.pattern+offset-k);
                uint64_t* text_blocks = (uint64_t*)(alignment.text+offset);
                uint64_t cmp = *pattern_blocks ^ *text_blocks;
                while (__builtin_expect(cmp==0,0)) {
                    offset += 8;
                    ++pattern_blocks;
                    ++text_blocks;
                    cmp = *pattern_blocks ^ *text_blocks;
                }
                // Count equal characters
                int equal_right_bits = __builtin_ctzl(cmp);
                int equal_chars = equal_right_bits/8;
                offset += equal_chars;
                // Return extended offset
                wf.mwavefronts[score_mod].offsets[k] = offset;
            }

            // Select end component
            switch (component_end) {
                case matrix_M: {
                    // Check diagonal/offset
                    if (wf.mwavefronts[score_mod].lo > alignment_k || alignment_k > wf.mwavefronts[score_mod].hi) end_reached = 0; // Not done
                    else{
                        int16_t moffset = wf.mwavefronts[score_mod].offsets[alignment_k];
                        if (moffset < alignment_offset) end_reached = 0; // Not done
                        else end_reached = 1;
                    }
                    break;
                }
                case matrix_I1: {
                    // Fetch I1-wavefront & check diagonal/offset
                    if (wf.iwavefronts[score_mod].offsets == NULL || wf.iwavefronts[score_mod].lo > alignment_k || alignment_k > wf.iwavefronts[score_mod].hi) end_reached = 0; // Not done
                    else{
                        int16_t i1offset = wf.iwavefronts[score_mod].offsets[alignment_k];
                        if (i1offset < alignment_offset) end_reached = 0; // Not done
                        else end_reached = 1;
                    }
                    break;
                }
                
                case matrix_D1: {
                    // Fetch D1-wavefront & check diagonal/offset
                    if (wf.dwavefronts[score_mod].offsets == NULL || wf.dwavefronts[score_mod].lo > alignment_k || alignment_k > wf.dwavefronts[score_mod].hi) end_reached = 0; // Not done
                    else{
                        int16_t d1offset = wf.dwavefronts[score_mod].offsets[alignment_k];
                        if (d1offset < alignment_offset) end_reached = 0; // Not done
                        else end_reached = 1;
                    }
                    break;
                }
                default:
                break;
            }
            if(end_reached){
                finish = 1;
            }else finish = 0;
        }
        

        if(finish){
            printf("allineamento finito\nScore: %d\n", score);
            printf("wf.mwavefronts.bt_pcigar[alignment_end_k] %d\n", wf.mwavefronts[score_mod].bt_pcigar[alignment_k]);
            break;
        }
        
        // Compute (s+1)-wavefront
        ++score;
        alignment.score = score;
        score_mod = score%max_score_scope;
        //wavefront compute affine
        int mism = score - penalties.mismatch;
        int gapopen = score - penalties.gap_open - penalties.gap_ext;
        int gapextend = score - penalties.gap_ext;

        int mismatch = mism % max_score_scope;
        int gap_open = gapopen % max_score_scope;
        int gap_extend = gapextend % max_score_scope;

        wf_t_cpu in_mwavefront_misms = (mismatch < 0 || wf.mwavefronts[mismatch].offsets == NULL || wf.mwavefronts[mismatch].null) ? wf.wavefront_null : wf.mwavefronts[mismatch];
        wf_t_cpu in_mwavefront_open = (gap_open < 0 || wf.mwavefronts[gap_open].offsets == NULL || wf.mwavefronts[gap_open].null) ? wf.wavefront_null : wf.mwavefronts[gap_open];
        wf_t_cpu in_iwavefront_ext = (gap_extend < 0 || wf.iwavefronts[gap_extend].offsets == NULL || wf.iwavefronts[gap_extend].null) ? wf.wavefront_null : wf.iwavefronts[gap_extend];
        wf_t_cpu in_dwavefront_ext = (gap_extend < 0 || wf.dwavefronts[gap_extend].offsets == NULL || wf.dwavefronts[gap_extend].null) ? wf.wavefront_null : wf.dwavefronts[gap_extend];
        
        if (in_mwavefront_misms.null && in_mwavefront_open.null && in_iwavefront_ext.null && in_dwavefront_ext.null) {
            alignment.num_null_steps++; // Increment null-steps
            // Nullify Wavefronts
            wf.mwavefronts[score_mod].null = 1;
            wf.iwavefronts[score_mod].null = 1;
            wf.dwavefronts[score_mod].null = 1;
        }else{

            alignment.num_null_steps = 0;

            int hi, lo;
            // Init
            int min_lo = in_mwavefront_misms.lo;
            int max_hi = in_mwavefront_misms.hi;
            
            if (min_lo > in_mwavefront_open.lo-1 && !in_mwavefront_open.null) min_lo = in_mwavefront_open.lo-1;
            if (max_hi < in_mwavefront_open.hi+1 && !in_mwavefront_open.null) max_hi = in_mwavefront_open.hi+1;
            if (min_lo > in_iwavefront_ext.lo+1 && !in_iwavefront_ext.null) min_lo = in_iwavefront_ext.lo+1;
            if (max_hi < in_iwavefront_ext.hi+1 && !in_iwavefront_ext.null) max_hi = in_iwavefront_ext.hi+1;
            if (min_lo > in_dwavefront_ext.lo-1 && !in_dwavefront_ext.null) min_lo = in_dwavefront_ext.lo-1;
            if (max_hi < in_dwavefront_ext.hi-1 && !in_dwavefront_ext.null) max_hi = in_dwavefront_ext.hi-1;
            lo = min_lo;
            hi = max_hi;
            
            int effective_lo = lo;
            int effective_hi = hi;
            int eff_lo = effective_lo - (max_score_scope + 1);
            int eff_hi = effective_hi + (max_score_scope + 1);
            effective_lo = MIN(eff_lo,alignment.historic_min_lo);
            effective_hi = MAX(eff_hi,alignment.historic_max_hi);
            
            // Allocate M-Wavefront
            alignment.historic_min_lo = effective_lo;
            alignment.historic_max_hi = effective_hi;

            wf.mwavefronts[score_mod].lo = lo;
            wf.mwavefronts[score_mod].hi = hi;

            // Allocate I1-Wavefront            
            if (!in_mwavefront_open.null || !wf.iwavefronts[gap_extend].null) {
                wf.iwavefronts[score_mod].lo = lo;
                wf.iwavefronts[score_mod].hi = hi;
            } else {
                wf.iwavefronts[score_mod].null = 1;
            }
            // Allocate D1-Wavefront
            if (!in_mwavefront_open.null || !wf.dwavefronts[gap_extend].null) {
                wf.dwavefronts[score_mod].lo = lo;
                wf.dwavefronts[score_mod].hi = hi;
            } else {
                wf.dwavefronts[score_mod].null = 1;
            }
            
            // Init missing elements, instead of loop peeling (M)
            bool m_misms_null = in_mwavefront_misms.null;
            bool m_gap_null = in_mwavefront_open.null;
            bool i_ext_null = in_iwavefront_ext.null;
            bool d_ext_null = in_dwavefront_ext.null;
            
            if (!m_misms_null) {
                //in_mwavefront_misms.offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
                
                if (in_mwavefront_misms.wf_elements_init_max >= hi){ 
                }else{
                    // Initialize lower elements
                    int max_init = MAX(in_mwavefront_misms.wf_elements_init_max,in_mwavefront_misms.hi);
                    int k;
                    for (k=max_init+1;k<=hi;++k) {
                        in_mwavefront_misms.offsets[k] = OFFSET_NULL;
                        in_mwavefront_misms.bt_prev[k] = 0;
                        in_mwavefront_misms.bt_pcigar[k] = 0;
                    }
                    // Set new maximum
                    in_mwavefront_misms.wf_elements_init_max = hi;
                }   

                //------------------------------------------------------------------------------------//

                if (in_mwavefront_misms.wf_elements_init_min <= lo){
                }else{  
                    // Initialize lower elements
                    int min_init = MIN(in_mwavefront_misms.wf_elements_init_min,in_mwavefront_misms.lo);
                    int k;
                    for (k=lo;k<min_init;++k) {
                        in_mwavefront_misms.offsets[k] = OFFSET_NULL;
                        in_mwavefront_misms.bt_prev[k] = 0;
                        in_mwavefront_misms.bt_pcigar[k] = 0;
                    }
                    // Set new minimum
                    in_mwavefront_misms.wf_elements_init_min = lo;
                }   
            }
            if (!m_gap_null) {
                //wf.mwavefronts[gap_open].offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
                if (in_mwavefront_open.wf_elements_init_max >= hi+1){
                }else{
                    // Initialize lower elements
                    int max_init = MAX(in_mwavefront_open.wf_elements_init_max,in_mwavefront_open.hi);
                    int k;
                    for (k=max_init+1;k<=hi+1;++k) {
                        in_mwavefront_open.offsets[k] = OFFSET_NULL;
                        in_mwavefront_open.bt_prev[k] = 0;                          
                        in_mwavefront_open.bt_pcigar[k] = 0;      
                    }
                    // Set new maximum
                    in_mwavefront_open.wf_elements_init_max = hi+1;

                }
                
                //------------------------------------------------------------------------------------//

                if (in_mwavefront_open.wf_elements_init_min <= lo-1){
                }else{
                    // Initialize lower elements
                    int min_init = MIN(in_mwavefront_open.wf_elements_init_min, in_mwavefront_open.lo);
                    int k;
                    for (k=lo-1;k<min_init;++k) {
                        in_mwavefront_open.offsets[k] = OFFSET_NULL;
                        in_mwavefront_open.bt_prev[k] = 0;
                        in_mwavefront_open.bt_pcigar[k] = 0;
                    }
                    // Set new minimum
                    in_mwavefront_open.wf_elements_init_min = lo-1;
                }
            }
            if (!i_ext_null) {
                //in_iwavefront_ext.offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
                if (in_iwavefront_ext.wf_elements_init_max >= hi){
                }else{
                    // Initialize lower elements
                    int max_init = MAX(in_iwavefront_ext.wf_elements_init_max,in_iwavefront_ext.hi);
                    int k;
                    for (k=max_init+1;k<=hi;++k) {
                        in_iwavefront_ext.offsets[k] = OFFSET_NULL;
                        in_iwavefront_ext.bt_prev[k] = 0;
                        in_iwavefront_ext.bt_pcigar[k] = 0;
                    }
                    // Set new maximum
                    in_iwavefront_ext.wf_elements_init_max = hi;
                }
                
                //------------------------------------------------------------------------------------//
                if (in_iwavefront_ext.wf_elements_init_min <= lo-1){
                }else{
                    // Initialize lower elements
                    int min_init = MIN(in_iwavefront_ext.wf_elements_init_min,in_iwavefront_ext.lo);
                    int k;
                    for (k=lo-1;k<min_init;++k) {
                        in_iwavefront_ext.offsets[k] = OFFSET_NULL;
                        in_iwavefront_ext.bt_prev[k] = 0;
                        in_iwavefront_ext.bt_pcigar[k] = 0;
                    }
                    // Set new minimum
                    in_iwavefront_ext.wf_elements_init_min = lo-1;
                }       
            }

            if (!d_ext_null) {
                //in_dwavefront_ext.offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
                if (in_dwavefront_ext.wf_elements_init_max >= hi+1){
                }else{
                    // Initialize lower elements
                    int16_t* offsets = in_dwavefront_ext.offsets;
                    int max_init = MAX(in_dwavefront_ext.wf_elements_init_max,in_dwavefront_ext.hi);
                    int k;
                    for (k=max_init+1;k<=hi+1;++k) {
                        in_dwavefront_ext.offsets[k] = OFFSET_NULL;
                        in_dwavefront_ext.bt_prev[k] = 0;
                        in_dwavefront_ext.bt_pcigar[k] = 0;
                    }
                    // Set new maximum
                    in_dwavefront_ext.wf_elements_init_max = hi+1;
                }

                //------------------------------------------------------------------------------------//
                if (in_dwavefront_ext.wf_elements_init_min <= lo){
                }else{
                    // Initialize lower elements
                    int16_t* offsets = in_dwavefront_ext.offsets;
                    int min_init = MIN(in_dwavefront_ext.wf_elements_init_min,in_dwavefront_ext.lo);
                    int k;
                    for (k=lo;k<min_init;++k) {
                        in_dwavefront_ext.offsets[k] = OFFSET_NULL;
                        in_dwavefront_ext.bt_prev[k] = 0;
                        in_dwavefront_ext.bt_pcigar[k] = 0;
                    }
                    // Set new minimum
                    in_dwavefront_ext.wf_elements_init_min = lo;
                }
            }
            
            
            // In Offsets
            wf_length = hi - lo + 1;
            wf.mwavefronts[score_mod].offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
            wf.iwavefronts[score_mod].offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
            wf.dwavefronts[score_mod].offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
            wf.mwavefronts[score_mod].bt_pcigar = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
            wf.iwavefronts[score_mod].bt_pcigar = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
            wf.dwavefronts[score_mod].bt_pcigar = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
            wf.mwavefronts[score_mod].bt_prev = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
            wf.iwavefronts[score_mod].bt_prev = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
            wf.dwavefronts[score_mod].bt_prev = (uint32_t*)malloc(sizeof(uint32_t)*wf_length);
            
            int16_t* misms_m = in_mwavefront_misms.offsets;
            int16_t* open_m = in_mwavefront_open.offsets;
            int16_t* ext_i = in_iwavefront_ext.offsets;
            int16_t* ext_d = in_dwavefront_ext.offsets;

            uint32_t* m_misms_bt_pcigar = in_mwavefront_misms.bt_pcigar;
            uint32_t* m_open_bt_pcigar = in_mwavefront_open.bt_pcigar;
            uint32_t* i_ext_bt_pcigar  = in_iwavefront_ext.bt_pcigar;
            uint32_t* d_ext_bt_pcigar  = in_dwavefront_ext.bt_pcigar;
            // In BT-prev
            uint32_t* m_misms_bt_prev = in_mwavefront_misms.bt_prev;
            uint32_t* m_open_bt_prev = in_mwavefront_open.bt_prev;
            uint32_t* i_ext_bt_prev  = in_iwavefront_ext.bt_prev;
            uint32_t* d_ext_bt_prev  = in_dwavefront_ext.bt_prev;

            // Compute-Next kernel loop
            int k;
            for (k=lo;k<=hi;++k) {
                // Update I1
                int16_t ins_o = open_m[k-1];
                int16_t ins_e = ext_i[k-1];
                int16_t ins;
                uint32_t ins_pcigar;
                uint32_t ins_block_idx;
                if (ins_e >= ins_o) {
                    ins = ins_e + 1;
                    ins_pcigar = i_ext_bt_pcigar[k-1];
                    ins_block_idx = i_ext_bt_prev[k-1];
                } else {
                    ins = ins_o + 1;
                    ins_pcigar = m_open_bt_pcigar[k-1];
                    ins_block_idx = m_open_bt_prev[k-1];
                }
                wf.iwavefronts[score_mod].bt_pcigar[k] = PCIGAR_PUSH_BACK_INS(ins_pcigar);
                wf.iwavefronts[score_mod].bt_prev[k] = ins_block_idx;
                wf.iwavefronts[score_mod].offsets[k] = ins;
                // Update D1
                int16_t  del_o = open_m[k+1];
                int16_t del_e = ext_d[k+1];
                int16_t del;
                uint32_t del_pcigar;
                uint32_t del_block_idx;
                if (del_e >= del_o) {
                    del = del_e;
                    del_pcigar = d_ext_bt_pcigar[k+1];
                    del_block_idx = d_ext_bt_prev[k+1];
                } else {
                    del = del_o;
                    del_pcigar = m_open_bt_pcigar[k+1];
                    del_block_idx = m_open_bt_prev[k+1];
                }
                wf.dwavefronts[score_mod].bt_pcigar[k] = PCIGAR_PUSH_BACK_DEL(del_pcigar);
                wf.dwavefronts[score_mod].bt_prev[k] = del_block_idx;
                wf.dwavefronts[score_mod].offsets[k] = del;

                // Update M
                int16_t misms = misms_m[k] + 1;
                int16_t max = MAX(del,MAX(misms,ins));
                if (max == ins) {
                    wf.mwavefronts[score_mod].bt_pcigar[k] = wf.iwavefronts[score_mod].bt_pcigar[k];
                    wf.mwavefronts[score_mod].bt_prev[k] = wf.iwavefronts[score_mod].bt_prev[k];
                }
                if (max == del) {
                    wf.mwavefronts[score_mod].bt_pcigar[k] = wf.dwavefronts[score_mod].bt_pcigar[k];
                    wf.mwavefronts[score_mod].bt_prev[k] = wf.dwavefronts[score_mod].bt_prev[k];
                }
                if (max == misms) {
                    wf.mwavefronts[score_mod].bt_pcigar[k] = PCIGAR_PUSH_BACK_MISMS(m_misms_bt_pcigar[k]);
                    wf.mwavefronts[score_mod].bt_prev[k] = m_misms_bt_prev[k];
                }


                // Coming from I/D -> X is fake to represent gap-close
                // Coming from M -> X is real to represent mismatch
                //wf.mwavefronts[score_mod].bt_pcigar[k] = PCIGAR_PUSH_BACK_MISMS(wf.mwavefronts[score_mod].bt_pcigar[k]);
                // Adjust offset out of boundaries !(h>tlen,v>plen) (here to allow vectorization)
                uint16_t h = max; 
                uint16_t v = max-k; 
                if (h > text_len) max = OFFSET_NULL;
                if (v > pattern_len) max = OFFSET_NULL;
                wf.mwavefronts[score_mod].offsets[k] = max;
            }
            printf("M PCIGAR: ");
            for(int i=lo; i<=hi; i++){
                printf("%d ", wf.mwavefronts[score_mod].bt_pcigar[i]);
            }
            printf("\n");

            printf("M PREV: ");
            for(int i=lo; i<=hi; i++){
                printf("%d ", wf.mwavefronts[score_mod].bt_prev[i]);
            }
            printf("\n");

            printf("I PCIGAR: ");
            for(int i=lo; i<=hi; i++){
                printf("%d ", wf.iwavefronts[score_mod].bt_pcigar[i]);
            }
            printf("\n");

            printf("I PREV: ");
            for(int i=lo; i<=hi; i++){
                printf("%d ", wf.iwavefronts[score_mod].bt_prev[i]);
            }
            printf("\n");

            printf("D PCIGAR: ");
            for(int i=lo; i<=hi; i++){
                printf("%d ", wf.dwavefronts[score_mod].bt_pcigar[i]);
            }
            printf("\n");

            printf("D PREV: ");
            for(int i=lo; i<=hi; i++){
                printf("%d ", wf.dwavefronts[score_mod].bt_prev[i]);
            }
            printf("\n");

            //wavefront_compute_process_ends(wf_aligner,&wavefront_set,score);
            // Trim ends from non-null WFs
            if (wf.mwavefronts[score_mod].offsets){
                int k;
                int lo = wf.mwavefronts[score_mod].lo;
                for (k=wf.mwavefronts[score_mod].hi;k>=lo;--k) {
                    // Fetch offset
                    int16_t offset = wf.mwavefronts[score_mod].offsets[k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                wf.mwavefronts[score_mod].hi = k; // Set new hi
                wf.mwavefronts[score_mod].wf_elements_init_max = k;
                // Trim from lo
                int hi = wf.mwavefronts[score_mod].hi;
                for (k=wf.mwavefronts[score_mod].lo;k<=hi;++k) {
                    // Fetch offset
                    int16_t offset = wf.mwavefronts[score_mod].offsets[k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                wf.mwavefronts[score_mod].lo = k; // Set new lo
                wf.mwavefronts[score_mod].wf_elements_init_min = k;
                wf.mwavefronts[score_mod].null = (wf.mwavefronts[score_mod].lo > wf.mwavefronts[score_mod].hi);
            }

            if (wf.iwavefronts[score_mod].offsets){
                int k;
                int lo = wf.iwavefronts[score_mod].lo;
                for (k=wf.iwavefronts[score_mod].hi;k>=lo;--k) {
                    // Fetch offset
                    int16_t offset = wf.iwavefronts[score_mod].offsets[k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                wf.iwavefronts[score_mod].hi = k; // Set new hi
                wf.iwavefronts[score_mod].wf_elements_init_max = k;
                // Trim from lo
                int hi = wf.iwavefronts[score_mod].hi;
                for (k=wf.iwavefronts[score_mod].lo;k<=hi;++k) {
                    // Fetch offset
                    int16_t offset = wf.iwavefronts[score_mod].offsets[k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                wf.iwavefronts[score_mod].lo = k; // Set new lo
                wf.iwavefronts[score_mod].wf_elements_init_min = k;
                wf.iwavefronts[score_mod].null = (wf.iwavefronts[score_mod].lo > wf.iwavefronts[score_mod].hi);
            }

            if (wf.dwavefronts[score_mod].offsets){
                int k;
                int lo = wf.dwavefronts[score_mod].lo;
                for (k=wf.dwavefronts[score_mod].hi;k>=lo;--k) {
                    // Fetch offset
                    int16_t offset = wf.dwavefronts[score_mod].offsets[k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                wf.dwavefronts[score_mod].hi = k; // Set new hi
                wf.dwavefronts[score_mod].wf_elements_init_max = k;
                // Trim from lo
                int hi = wf.dwavefronts[score_mod].hi;
                for (k=wf.dwavefronts[score_mod].lo;k<=hi;++k) {
                    // Fetch offset
                    int16_t offset = wf.dwavefronts[score_mod].offsets[k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                wf.dwavefronts[score_mod].lo = k; // Set new lo
                wf.dwavefronts[score_mod].wf_elements_init_min = k;
                wf.dwavefronts[score_mod].null = (wf.dwavefronts[score_mod].lo > wf.dwavefronts[score_mod].hi);
            }
        }
    }

    printf("finished correctly\n");

    return 0;
}


