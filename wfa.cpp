#include <cstdlib>
#include <string>
#include <vector>
#include <cstring>
#include "common.h"
#include <omp.h>

extern "C" {
	#include "wavefront/wavefront_align.h"
}


int main(int argc, char const *argv[]){

    int num_couples;
    penalties_t penalties;
    bool check;

    if(argc!=6){
		printf("ERROR! Please specify in order: mismatch, gap_opening, gap_extension, file name, check\n");
		return 0;
	}

	penalties.mismatch = atoi(argv[1]);
	penalties.gap_open = atoi(argv[2]);
	penalties.gap_ext = atoi(argv[3]);
    penalties.match = 0; 
    penalties.gap_open2 = -1;
    penalties.gap_ext2 = -1;
    FILE* fp = fopen(argv[4], "r");
    check = atoi(argv[5]);

    if(fp==NULL){
		printf("ERROR: Cannot open file.\n");
		return 1;
	}
    int error_rate;
    int seq_len;
    fscanf(fp, "%d %d %d", &num_couples, &seq_len, &error_rate);
    int pattern_len = seq_len;
    int text_len = seq_len;
    
    wf_t alignment; 
    wf_components_t wf;

    alignment.pattern = (char*)malloc(sizeof(char)*pattern_len);
    alignment.text = (char*)malloc(sizeof(char)*text_len);
    for(int i = 0; i<num_couples; i++){
		fscanf(fp, "%s", alignment.pattern+(i*pattern_len));
		fscanf(fp, "%s", alignment.text+(i*text_len));
	}
    alignment.score = 0;
    alignment.wf_elements_init_max = 0;
    alignment.wf_elements_init_min = 0;

    int max_score = MAX(penalties.gap_open+penalties.gap_ext, penalties.mismatch) + 1;
    int abs_seq_diff = ABS(pattern_len-text_len);
    int max_score_misms = MIN(pattern_len,text_len) * penalties.mismatch;
    int max_score_indel = penalties.gap_open + abs_seq_diff * penalties.gap_ext;
    int num_wavefronts = max_score_misms + max_score_indel + 1;
    int max_op = 2*(pattern_len+text_len);
    matrix_type component_end = matrix_M;

    int hi = 0;
    int lo = 0;
    int eff_lo = lo - (max_score + 1);
    int eff_hi = hi + (max_score + 1);
    lo = MIN(eff_lo,0);
    hi = MAX(eff_hi,0);
    int wf_length = hi - lo + 1;

    wf.mwavefronts = (wf_t*)malloc(sizeof(wf_t)*num_wavefronts);
    wf.dwavefronts = (wf_t*)malloc(sizeof(wf_t)*num_wavefronts);
    wf.iwavefronts = (wf_t*)malloc(sizeof(wf_t)*num_wavefronts);
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

    for(int i=-(wf_length/2); i<(wf_length/2); i++){
        wf.wavefront_null.offsets[i] = OFFSET_NULL;
    }
    
    alignment.num_null_steps = 0;
    alignment.historic_max_hi = 0;
    alignment.historic_min_lo = 0;
    int score = alignment.score;
    wf.mwavefronts[score].lo = 0;
    wf.mwavefronts[score].hi = 0;
    bool finish = 0;

    int alignment_k = text_len-pattern_len;
    int16_t alignment_offset = text_len;
    
    t_cigar cigar;
    cigar.begin_offset = 0;
    cigar.end_offset = 0;
    cigar.score = INT16_MIN;
    cigar.operations = (char*)malloc(sizeof(char)*max_op);
    
    while (true) {
        // Exact extend s-wavefront
        // Fetch m-wavefront
        if (wf.mwavefronts[score].offsets == NULL) {
            if(alignment.num_null_steps > max_score){
                finish = 1; //done
            }else{
                finish = 0; // not done
            }

        }else{
            bool end_reached = false;
            int k;
            for (k=wf.mwavefronts[score].lo;k<=wf.mwavefronts[score].hi;++k) {
                
                int16_t offset = wf.mwavefronts[score].offsets[k]; //offset + k
                if (offset == OFFSET_NULL) {continue;}
                uint64_t* pattern_blocks = (uint64_t*)(alignment.pattern+offset-k);
                uint64_t* text_blocks = (uint64_t*)(alignment.text+offset);
                
                uint64_t cmp = (*pattern_blocks) ^ (*text_blocks);
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
                wf.mwavefronts[score].offsets[k] = offset;
            }

            // Select end component
            switch (component_end) {
                case matrix_M: {
                    // Check diagonal/offset
                    if (wf.mwavefronts[score].lo > alignment_k || alignment_k > wf.mwavefronts[score].hi) {
                        end_reached = 0; // Not done
                    }else{
                        int16_t moffset = wf.mwavefronts[score].offsets[alignment_k];
                        if (moffset < alignment_offset) end_reached = 0; // Not done
                        else {end_reached = 1;}
                    }
                    break;
                }
                case matrix_I1: {
                    // Fetch I1-wavefront & check diagonal/offset
                    if (wf.iwavefronts[score].offsets == NULL || wf.iwavefronts[score].lo > alignment_k || alignment_k > wf.iwavefronts[score].hi) end_reached = 0; // Not done
                    else{
                        int16_t i1offset = wf.iwavefronts[score].offsets[alignment_k];
                        if (i1offset < alignment_offset) end_reached = 0; // Not done
                        else {end_reached = 1;}
                    }
                    break;
                }
                
                case matrix_D1: {
                    // Fetch D1-wavefront & check diagonal/offset
                    if (wf.dwavefronts[score].offsets == NULL || wf.dwavefronts[score].lo > alignment_k || alignment_k > wf.dwavefronts[score].hi) end_reached = 0; // Not done
                    else{
                        int16_t d1offset = wf.dwavefronts[score].offsets[alignment_k];
                        if (d1offset < alignment_offset) end_reached = 0; // Not done
                        else {end_reached = 1;}
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
            alignment.score = score;
            // Prepare cigar
            cigar.end_offset = max_op - 1;
            cigar.begin_offset = max_op - 2;
            cigar.operations[cigar.end_offset] = '\0';
            
            matrix_type matrix_type = matrix_M;
            int k = alignment_k;
            int h = alignment_offset;
            int v = alignment_offset - alignment_k;
            int16_t offset = alignment_offset;
            // Account for ending insertions/deletions
            if (component_end == matrix_M) {
                if (v < pattern_len) {
                    int i = pattern_len - v;
                    while (i > 0) {cigar.operations[cigar.begin_offset--] = 'D'; --i;};
                }
                if (h < text_len) {
                    int i = text_len - h;
                    while (i > 0) {cigar.operations[cigar.begin_offset--] = 'I'; --i;};                                                                              
                }
            }
            
            // Trace the alignment back
            while (v > 0 && h > 0 && score > 0) {
                int mismatch = score - penalties.mismatch;
                int gap_open = score - penalties.gap_open - penalties.gap_ext;
                int gap_extend = score - penalties.gap_ext;
                int gap_open2 = score - penalties.gap_open2 - penalties.gap_ext2;
                int gap_extend2 = score - penalties.gap_ext2;
                int64_t max_all;
                
                switch (matrix_type) {
                    case matrix_M: {
                        
                        int64_t misms;
                        if (mismatch < 0) misms = OFFSET_NULL; 
                        else{
                            if (!wf.mwavefronts[mismatch].null && wf.mwavefronts[mismatch].lo <= k && k <= wf.mwavefronts[mismatch].hi) {
                                misms = ((((int64_t)(wf.mwavefronts[mismatch].offsets[k]+1)) << 4) | bt_M);
                            } else {
                                misms = OFFSET_NULL;
                            }
                        }

                        int64_t ins_open;
                        if (gap_open < 0) ins_open = OFFSET_NULL;
                        else{
                            if (!wf.mwavefronts[gap_open].null && wf.mwavefronts[gap_open].lo <= k-1 && k-1 <= wf.mwavefronts[gap_open].hi) {
                                ins_open = ((((int64_t)(wf.mwavefronts[gap_open].offsets[k-1]+1)) << 4) | bt_I1_open);
                            } else {
                                ins_open = OFFSET_NULL;
                            }
                        }

                        int64_t ins_ext;
                        if (gap_extend < 0) ins_ext = OFFSET_NULL;
                        else{
                            if (!wf.iwavefronts[gap_extend].null && wf.iwavefronts[gap_extend].lo <= k-1 && k-1 <= wf.iwavefronts[gap_extend].hi) {
                                ins_ext = ((((int64_t)(wf.iwavefronts[gap_extend].offsets[k-1]+1)) << 4) | bt_I1_ext);
                            } else {
                                ins_ext = OFFSET_NULL;
                            }
                        }
                        
                        int64_t max_ins = MAX(ins_open,ins_ext);
                        int64_t del_open;
                        if (gap_open < 0) del_open = OFFSET_NULL;
                        else{
                            if (!wf.mwavefronts[gap_open].null && wf.mwavefronts[gap_open].lo <= k+1 && k+1 <= wf.mwavefronts[gap_open].hi) {
                                del_open = ((((int64_t)(wf.mwavefronts[gap_open].offsets[k+1])) << 4) | bt_D1_open);
                            } else {
                                del_open = OFFSET_NULL;
                            }
                        }

                        int64_t del_ext;
                        if (gap_extend < 0) del_ext = OFFSET_NULL;
                        else{
                            if (!wf.dwavefronts[gap_extend].null && wf.dwavefronts[gap_extend].lo <= k+1 && k+1 <= wf.dwavefronts[gap_extend].hi) {
                                del_ext = ((((int64_t)(wf.dwavefronts[gap_extend].offsets[k+1])) << 4) | bt_D1_ext);
                            } else {
                                del_ext = OFFSET_NULL;
                            }
                        }
                        int64_t max_del = MAX(del_open,del_ext);
                        max_all = MAX(misms,MAX(max_ins,max_del));
                        break;
                    }

                    case matrix_I1: {
                        int64_t ins_open;
                        if (gap_open < 0) ins_open = OFFSET_NULL;
                        if (!wf.mwavefronts[gap_open].null && wf.mwavefronts[gap_open].lo <= k-1 && k-1 <= wf.mwavefronts[gap_open].hi) {
                            ins_open = ((((int64_t)(wf.mwavefronts[gap_open].offsets[k-1]+1)) << 4) | bt_I1_open);
                        } else {
                            ins_open = OFFSET_NULL;
                        }

                        int64_t ins_ext;
                        if (gap_extend < 0) ins_ext = OFFSET_NULL;
                        if (!wf.iwavefronts[gap_extend].null && wf.iwavefronts[gap_extend].lo <= k-1 && k-1 <= wf.iwavefronts[gap_extend].hi) {
                            ins_ext = ((((int64_t)(wf.iwavefronts[gap_extend].offsets[k-1]+1)) << 4) | bt_I1_ext);
                        } else {
                            ins_ext = OFFSET_NULL;
                        }

                        max_all = MAX(ins_open,ins_ext);
                        break;
                    }

                    /* case matrix_I2: {
                        int64_t ins2_open = wavefront_backtrace_ins2_open(wf_aligner,gap_open2,k);
                        int64_t ins2_ext  = wavefront_backtrace_ins2_ext(wf_aligner,gap_extend2,k);
                        max_all = MAX(ins2_open,ins2_ext);
                        break;
                    } */

                    case matrix_D1: {
                        int64_t del_open;
                        if (gap_open < 0) del_open = OFFSET_NULL;
                        if (!wf.mwavefronts[gap_open].null && wf.mwavefronts[gap_open].lo <= k+1 && k+1 <= wf.mwavefronts[gap_open].hi) {
                            del_open = ((((int64_t)(wf.mwavefronts[gap_open].offsets[k+1])) << 4) | bt_D1_open);
                        } else {
                            del_open = OFFSET_NULL;
                        }

                        int64_t del_ext;
                        if (gap_extend < 0) del_ext = OFFSET_NULL;
                        if (!wf.dwavefronts[gap_extend].null && wf.dwavefronts[gap_extend].lo <= k+1 && k+1 <= wf.dwavefronts[gap_extend].hi) {
                            del_ext = ((((int64_t)(wf.dwavefronts[gap_extend].offsets[k+1])) << 4) | bt_D1_ext);
                        } else {
                            del_ext = OFFSET_NULL;
                        }

                        max_all = MAX(del_open,del_ext);
                        break;
                    }

                    /* case matrix_D2: {
                        const int64_t del2_open = wavefront_backtrace_del2_open(wf_aligner,gap_open2,k);
                        const int64_t del2_ext  = wavefront_backtrace_del2_ext(wf_aligner,gap_extend2,k);
                        max_all = MAX(del2_open,del2_ext);
                        break;
                    } */
                    
                    default:
                        fprintf(stderr,"[WFA::Backtrace] Wrong type trace.1\n");
                        exit(1);
                        break;
                }

                //wavefront_bt_Matches(wf_aligner,k,offset,num_matches,cigar);
                
                if (max_all < 0) break; // No source
                // Traceback Matches
                if (matrix_type == matrix_M) {
                    int max_offset = ((max_all) >> 4);
                    int num_matches = offset - max_offset;
                    const uint64_t matches_lut = 0x4D4D4D4D4D4D4D4Dul; // Matches LUT = "MMMMMMMM"
                    char* operations = cigar.operations + cigar.begin_offset;
                    // Update offset first
                    cigar.begin_offset -= num_matches;
                    // Blocks of 8-matches
                    while (num_matches >= 8) {
                        operations -= 8;
                        *((uint64_t*)(operations+1)) = matches_lut;
                        num_matches -= 8;
                    }
                    // Remaining matches
                    int i;
                    for (i=0;i<num_matches;++i) {
                        *operations = 'M';
                        --operations;
                    }

                    offset = max_offset;
                    // Update coordinates
                    v = offset-k;
                    h = offset;
                    if (v <= 0 || h <= 0) break;
                }
                // Traceback Operation

                bt_type_t backtrace_type = (bt_type_t)((max_all) & 0x000000000000000Fl);

                switch (backtrace_type) {
                case bt_M:
                    score = mismatch;
                    matrix_type = matrix_M;
                    break;
                case bt_I1_open:
                    score = gap_open;
                    matrix_type = matrix_M;
                    break;
                case bt_I1_ext:
                    score = gap_extend;
                    matrix_type = matrix_I1;
                    break;
                case bt_I2_open:
                    score = gap_open2;
                    matrix_type = matrix_M;
                    break;
                case bt_I2_ext:
                    score = gap_extend2;
                    matrix_type = matrix_I2;
                    break;
                case bt_D1_open:
                    score = gap_open;
                    matrix_type = matrix_M;
                    break;
                case bt_D1_ext:
                    score = gap_extend;
                    matrix_type = matrix_D1;
                    break;
                case bt_D2_open:
                    score = gap_open2;
                    matrix_type = matrix_M;
                    break;
                case bt_D2_ext:
                    score = gap_extend2;
                    matrix_type = matrix_D2;
                    break;
                default:
                    fprintf(stderr,"[WFA::Backtrace] Wrong type trace.2\n");
                    exit(1);
                    break;
                }


                switch (backtrace_type) {
                    case bt_M:
                        cigar.operations[cigar.begin_offset--] = 'X';
                        --offset;
                        break;
                    case bt_I1_open:
                    case bt_I1_ext:
                    case bt_I2_open:
                    case bt_I2_ext:
                        cigar.operations[cigar.begin_offset--] = 'I';
                        --k; 
                        --offset;
                        break;
                    case bt_D1_open:
                    case bt_D1_ext:
                    case bt_D2_open:
                    case bt_D2_ext:
                        cigar.operations[cigar.begin_offset--] = 'D';
                        ++k;
                        break;
                    default:
                        fprintf(stderr,"[WFA::Backtrace] Wrong type trace.3\n");
                        exit(1);
                        break;
                    }


                // Update coordinates
                v = offset-k;
                h = offset;
            }

            
            // Account for last operations
            if (matrix_type == matrix_M) {
                if (v > 0 && h > 0) {
                    // Account for beginning series of matches
                    int num_matches = MIN(v,h);
                    uint64_t matches_lut = 0x4D4D4D4D4D4D4D4Dul; // Matches LUT = "MMMMMMMM"
                    char* operations = cigar.operations + cigar.begin_offset;
                    // Update offset first
                    cigar.begin_offset -= num_matches;
                    // Blocks of 8-matches
                    while (num_matches >= 8) {
                        operations -= 8;
                        *((uint64_t*)(operations+1)) = matches_lut;
                        num_matches -= 8;
                    }
                    // Remaining matches
                    int i;
                    for (i=0;i<num_matches;++i) {
                        *operations = 'M';
                        --operations;
                    }
                    
                    
                    v -= num_matches;
                    h -= num_matches;
                }
                // Account for beginning insertions/deletions
                while (v > 0) {cigar.operations[cigar.begin_offset--] = 'D'; --v;};
                while (h > 0) {cigar.operations[cigar.begin_offset--] = 'I'; --h;};
            } 
            // Set CIGAR
            ++(cigar.begin_offset);
            cigar.score = score;
            printf("%s\n", cigar.operations);




            char* operations = cigar.operations;
            // Allocate alignment buffers
            int max_buffer_length = text_len+pattern_len+1;
            char* pattern_alg = (char*)malloc(sizeof(char)*max_buffer_length);
            char* ops_alg = (char*)malloc(sizeof(char)*max_buffer_length);
            char* text_alg = (char*)malloc(sizeof(char)*max_buffer_length);
            // Compute alignment buffers
            int i, alg_pos = 0, pattern_pos = 0, text_pos = 0;
            for (i=cigar.begin_offset;i<cigar.end_offset;++i) {
                switch (operations[i]) {
                case 'M':
                    if (alignment.pattern[pattern_pos] != alignment.text[text_pos]) {
                    pattern_alg[alg_pos] = alignment.pattern[pattern_pos];
                    ops_alg[alg_pos] = 'X';
                    text_alg[alg_pos++] = alignment.text[text_pos];
                    } else {
                    pattern_alg[alg_pos] = alignment.pattern[pattern_pos];
                    ops_alg[alg_pos] = '|';
                    text_alg[alg_pos++] = alignment.text[text_pos];
                    }
                    pattern_pos++; text_pos++;
                    break;
                case 'X':
                    if (alignment.pattern[pattern_pos] != alignment.text[text_pos]) {
                    pattern_alg[alg_pos] = alignment.pattern[pattern_pos++];
                    ops_alg[alg_pos] = ' ';
                    text_alg[alg_pos++] = alignment.text[text_pos++];
                    } else {
                    pattern_alg[alg_pos] = alignment.pattern[pattern_pos++];
                    ops_alg[alg_pos] = 'X';
                    text_alg[alg_pos++] = alignment.text[text_pos++];
                    }
                    break;
                case 'I':
                    pattern_alg[alg_pos] = '-';
                    ops_alg[alg_pos] = ' ';
                    text_alg[alg_pos++] = alignment.text[text_pos++];
                    break;
                case 'D':
                    pattern_alg[alg_pos] = alignment.pattern[pattern_pos++];
                    ops_alg[alg_pos] = ' ';
                    text_alg[alg_pos++] = '-';
                    break;
                default:
                    break;
                }
            }
            i=0;
            while (pattern_pos < pattern_len) {
                pattern_alg[alg_pos+i] = alignment.pattern[pattern_pos++];
                ops_alg[alg_pos+i] = '?';
                ++i;
            }
            i=0;
            while (text_pos < text_len) {
                text_alg[alg_pos+i] = alignment.text[text_pos++];
                ops_alg[alg_pos+i] = '?';
                ++i;
            }
            // Print alignment pretty
            printf("      ALIGNMENT\t");
            bool print_matches = true;
            if (cigar.begin_offset >= cigar.end_offset){

            }else{
                // Print operations
                char last_op = cigar.operations[cigar.begin_offset];
                int last_op_length = 1;
                int i;
                for (i=cigar.begin_offset+1;i<cigar.end_offset;++i) {
                    if (cigar.operations[i]==last_op) {
                    ++last_op_length;
                    } else {
                        if (print_matches || last_op != 'M') {
                            printf("%d%c",last_op_length,last_op);
                        }
                        last_op = cigar.operations[i];
                        last_op_length = 1;
                    }
                }
                if (print_matches || last_op != 'M') {
                    printf("%d%c",last_op_length,last_op);
                }
            }
            

            printf("\n");
            printf("      ALIGNMENT.COMPACT\t");

            print_matches = false;
            if (cigar.begin_offset >= cigar.end_offset){

            }else{
                // Print operations
                char last_op = cigar.operations[cigar.begin_offset];
                int last_op_length = 1;
                int i;
                for (i=cigar.begin_offset+1;i<cigar.end_offset;++i) {
                    if (cigar.operations[i]==last_op) {
                    ++last_op_length;
                    } else {
                        if (print_matches || last_op != 'M') {
                            printf("%d%c",last_op_length,last_op);
                        }
                        last_op = cigar.operations[i];
                        last_op_length = 1;
                    }
                }
                if (print_matches || last_op != 'M') {
                    printf("%d%c",last_op_length,last_op);
                }
            }
            
            printf("\n");
            printf("      PATTERN    %s\n",pattern_alg);
            printf("                 %s\n",ops_alg);
            printf("      TEXT       %s\n",text_alg);
            

            break;
        }
        
        // Compute (s+1)-wavefront
        ++score;        
        //wavefront compute affine
        int mismatch = score - penalties.mismatch;
        int gap_open = score - penalties.gap_open - penalties.gap_ext;
        int gap_extend = score - penalties.gap_ext;

        wf_t in_mwavefront_misms = (mismatch < 0 || wf.mwavefronts[mismatch].offsets == NULL || wf.mwavefronts[mismatch].null) ? wf.wavefront_null : wf.mwavefronts[mismatch];
        wf_t in_mwavefront_open = (gap_open < 0 || wf.mwavefronts[gap_open].offsets == NULL || wf.mwavefronts[gap_open].null) ? wf.wavefront_null : wf.mwavefronts[gap_open];
        wf_t in_iwavefront_ext = (gap_extend < 0 || wf.iwavefronts[gap_extend].offsets == NULL || wf.iwavefronts[gap_extend].null) ? wf.wavefront_null : wf.iwavefronts[gap_extend];
        wf_t in_dwavefront_ext = (gap_extend < 0 || wf.dwavefronts[gap_extend].offsets == NULL || wf.dwavefronts[gap_extend].null) ? wf.wavefront_null : wf.dwavefronts[gap_extend];
        
        if (in_mwavefront_misms.null && in_mwavefront_open.null && in_iwavefront_ext.null && in_dwavefront_ext.null) {
            alignment.num_null_steps++; // Increment null-steps
            // Nullify Wavefronts
            wf.mwavefronts[score].null = 1;
            wf.iwavefronts[score].null = 1;
            wf.dwavefronts[score].null = 1;
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
            int eff_lo = effective_lo - (max_score + 1);
            int eff_hi = effective_hi + (max_score + 1);
            effective_lo = MIN(eff_lo,alignment.historic_min_lo);
            effective_hi = MAX(eff_hi,alignment.historic_max_hi);
            
            // Allocate M-Wavefront
            alignment.historic_min_lo = effective_lo;
            alignment.historic_max_hi = effective_hi;

            wf.mwavefronts[score].lo = lo;
            wf.mwavefronts[score].hi = hi;
            // Allocate I1-Wavefront            
            if (!in_mwavefront_open.null || !wf.iwavefronts[gap_extend].null) {
                wf.iwavefronts[score].lo = lo;
                wf.iwavefronts[score].hi = hi;
            } else {
                wf.iwavefronts[score].null = 1;
            }
            // Allocate D1-Wavefront
            if (!in_mwavefront_open.null || !wf.dwavefronts[gap_extend].null) {
                wf.dwavefronts[score].lo = lo;
                wf.dwavefronts[score].hi = hi;
            } else {
                wf.dwavefronts[score].null = 1;
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
                    }
                    // Set new minimum
                    in_dwavefront_ext.wf_elements_init_min = lo;
                }
            }
            
            
            // In Offsets
            wf.mwavefronts[score].offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
            wf.iwavefronts[score].offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
            wf.dwavefronts[score].offsets = (int16_t*)malloc(sizeof(int16_t)*wf_length);
            int16_t* misms_m = in_mwavefront_misms.offsets;
            int16_t* open_m = in_mwavefront_open.offsets;
            int16_t* ext_i = in_iwavefront_ext.offsets;
            int16_t* ext_d = in_dwavefront_ext.offsets;

            // Compute-Next kernel loop
            for (int k=lo; k<=hi; ++k) {
                // Update I1
                
                int16_t ins_o = open_m[k-1];
                int16_t ins_e = ext_i[k-1];
                int16_t  ins = MAX(ins_o,ins_e) + 1;
                wf.iwavefronts[score].offsets[k] = ins;
                // Update D1
                int16_t  del_o = open_m[k+1];
                int16_t del_e = ext_d[k+1];
                int16_t del = MAX(del_o,del_e);
                wf.dwavefronts[score].offsets[k] = del;
                // Update M
                int16_t misms = misms_m[k] + 1;
                int16_t max = MAX(del,MAX(misms,ins));
                // Adjust offset out of boundaries !(h>tlen,v>plen) (here to allow vectorization)
                
                uint16_t h = max; 
                uint16_t v = max-k; 
                if (h > text_len) max = OFFSET_NULL;
                if (v > pattern_len) max = OFFSET_NULL;
                wf.mwavefronts[score].offsets[k] = max;
                
            }

            //wavefront_compute_process_ends(wf_aligner,&wavefront_set,score);
            // Trim ends from non-null WFs
    
            if (wf.mwavefronts[score].offsets){
                int k;
                int lo = wf.mwavefronts[score].lo;
                for (k=wf.mwavefronts[score].hi;k>=lo;--k) {
                    // Fetch offset
                    int16_t offset = wf.mwavefronts[score].offsets[k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                wf.mwavefronts[score].hi = k; // Set new hi
                wf.mwavefronts[score].wf_elements_init_max = k;
                // Trim from lo
                int hi = wf.mwavefronts[score].hi;
                for (k=wf.mwavefronts[score].lo;k<=hi;++k) {
                    // Fetch offset
                    int16_t offset = wf.mwavefronts[score].offsets[k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                wf.mwavefronts[score].lo = k; // Set new lo
                wf.mwavefronts[score].wf_elements_init_min = k;
                wf.mwavefronts[score].null = (wf.mwavefronts[score].lo > wf.mwavefronts[score].hi);
            }

            if (wf.iwavefronts[score].offsets){
                int k;
                int lo = wf.iwavefronts[score].lo;
                for (k=wf.iwavefronts[score].hi;k>=lo;--k) {
                    // Fetch offset
                    int16_t offset = wf.iwavefronts[score].offsets[k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                wf.iwavefronts[score].hi = k; // Set new hi
                
                wf.iwavefronts[score].wf_elements_init_max = k;
                // Trim from lo
                int hi = wf.iwavefronts[score].hi;
                for (k=wf.iwavefronts[score].lo;k<=hi;++k) {
                    // Fetch offset
                    int16_t offset = wf.iwavefronts[score].offsets[k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                wf.iwavefronts[score].lo = k; // Set new lo
                wf.iwavefronts[score].wf_elements_init_min = k;
                wf.iwavefronts[score].null = (wf.iwavefronts[score].lo > wf.iwavefronts[score].hi);
            }

            if (wf.dwavefronts[score].offsets){
                int k;
                int lo = wf.dwavefronts[score].lo;
                for (k=wf.dwavefronts[score].hi;k>=lo;--k) {
                    // Fetch offset
                    int16_t offset = wf.dwavefronts[score].offsets[k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                wf.dwavefronts[score].hi = k; // Set new hi
                wf.dwavefronts[score].wf_elements_init_max = k;
                // Trim from lo
                int hi = wf.dwavefronts[score].hi;
                for (k=wf.dwavefronts[score].lo;k<=hi;++k) {
                    // Fetch offset
                    int16_t offset = wf.dwavefronts[score].offsets[k];
                    // Check boundaries
                    uint32_t h = offset; // Make unsigned to avoid checking negative
                    uint32_t v = offset-k; // Make unsigned to avoid checking negative
                    if (h <= text_len && v <= pattern_len) break;
                }
                wf.dwavefronts[score].lo = k; // Set new lo
                wf.dwavefronts[score].wf_elements_init_min = k;
                wf.dwavefronts[score].null = (wf.dwavefronts[score].lo > wf.dwavefronts[score].hi);
            }
        }
    }

    //Confronto tool originale
    if(check){
        printf("ORIGINAL SW\n");
		// Configure alignment attributes
        wavefront_aligner_attr_t attributes = wavefront_aligner_attr_default;
        attributes.distance_metric = gap_affine;
        attributes.affine_penalties.mismatch = 4;
        attributes.affine_penalties.gap_opening = 6;
        attributes.affine_penalties.gap_extension = 2;
        // Initialize Wavefront Aligner
        wavefront_aligner_t* const wf_aligner = wavefront_aligner_new(&attributes);

        char *pattern, *text;
        pattern = (char*)malloc(sizeof(char)*pattern_len);
        text = (char*)malloc(sizeof(char)*text_len);
        strcpy(pattern, alignment.pattern);
        strcpy(text, alignment.text);
        
        wavefront_align(wf_aligner,pattern,strlen(pattern),text,strlen(text)); // Align

        // Display CIGAR & score
        cigar_print_pretty(stderr,pattern,strlen(pattern),text,strlen(text),
                        wf_aligner->cigar,wf_aligner->mm_allocator);

        printf("score sw %d\n", wf_aligner->cigar->score);
        if(wf_aligner->cigar->score == (-alignment.score))
            printf("\n\n |ALIGNEMENT FINISHED CORRECTLY|\n");
        else 
            printf("\n\n |ERROR|\n");

    }
    return 0;
}


