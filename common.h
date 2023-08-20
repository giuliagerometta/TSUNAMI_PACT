#ifndef COMMON_H
    #define COMMON_H
    
    #include <stdint.h>
    #define BLOCKDIM 128
    #define OFFSET_NULL (INT16_MIN/2)
    #define MIN(a,b) (((a)<=(b))?(a):(b))
    #define MAX(a,b) (((a)>=(b))?(a):(b))
    #define ABS(a) (((a)>=0)?(a):-(a))
    #define PCIGAR_MAX_LENGTH 16
    #define PATTERN_SIZE 32
    #define TEXT_SIZE 32
    #define WAVEFRONT_LENGTH(lo,hi) ((hi)-(lo)+1)
    #define WF_NULL_INIT_LO     (-1024)
    #define WF_NULL_INIT_HI     ( 1024)

    #define PCIGAR_DELETION    1ul
    #define PCIGAR_MISMATCH    2ul
    #define PCIGAR_INSERTION   3ul
    
    #define PCIGAR_PUSH_BACK_INS(pcigar)   ((pcigar<<2) | PCIGAR_INSERTION)
    #define PCIGAR_PUSH_BACK_DEL(pcigar)   ((pcigar<<2) | PCIGAR_DELETION)
    #define PCIGAR_PUSH_BACK_MISMS(pcigar) ((pcigar<<2) | PCIGAR_MISMATCH)

    typedef struct {
        char* operations;
        int max_operations;
        int begin_offset;
        int end_offset;
        int score;
    }t_cigar;
    
    typedef struct {
        int16_t* offsets; 
        char *pattern;
        char *text;
        int score; 
        int hi; 
        int lo; 
        int num_null_steps; 
        bool null; 
        int k; 
        int wf_elements_init_max;
        int wf_elements_init_min;
        int historic_min_lo;
        int historic_max_hi;
        t_cigar cigar;
        uint32_t* bt_pcigar;
        uint32_t* bt_prev;
    } wf_t;

    typedef enum {
        matrix_M = 0,
        matrix_I1 = 1,
        matrix_I2 = 2, 
        matrix_D1 = 3, 
        matrix_D2 = 4,
    } matrix_type;
    
    typedef struct {
        wf_t* mwavefronts;              // M-wavefronts
        wf_t* iwavefronts;              // I1-wavefronts
        wf_t* dwavefronts;              // D1-wavefronts
        wf_t wavefront_null;            // Null wavefront (orthogonal reading)
    } wf_components_t;

    typedef enum {
        bt_M       = 9,
        bt_D2_ext  = 8,
        bt_D2_open = 7,
        bt_D1_ext  = 6,
        bt_D1_open = 5,
        bt_I2_ext  = 4,
        bt_I2_open = 3,
        bt_I1_ext  = 2,
        bt_I1_open = 1,
    } bt_type_t;


    typedef struct{
        int match;
        int mismatch;
        int gap_open;
        int gap_ext;
        int gap_open2;
        int gap_ext2;
    }penalties_t;
    

#endif