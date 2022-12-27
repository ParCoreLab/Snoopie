#include <stdio.h>

// include nvbit read/write register device API
#include "nvbit_reg_rw.h"

extern "C" __device__ __noinline__ void mov_replace(int pred, int reg_dst_num,
                                                    int value_or_reg,
                                                    int is_op1_reg) {
    if (!pred) {
        return;
    }

    if (is_op1_reg) {
        if (is_op1_reg == 1) {
            /* read value of register source */
            int value = nvbit_read_reg(value_or_reg);
            /* write value in register destination */
            nvbit_write_reg(reg_dst_num, value);
        } else if (is_op1_reg == 2) {
            /* read value of uniform register source */
            int value = nvbit_read_ureg(value_or_reg);
            /* write value in register destination */
            nvbit_write_reg(reg_dst_num, value);
        }
    } else {
        /* immediate value, just write it in the register */
        nvbit_write_reg(reg_dst_num, value_or_reg);
    }
}
