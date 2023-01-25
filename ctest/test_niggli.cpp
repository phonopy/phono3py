#include <gtest/gtest.h>

extern "C" {
#include <math.h>

#include "niggli.h"
}

/**
 * @brief niggli:niggli_reduce
 * Run Niggli reduction.
 * @return succeeded (1) or not (0)
 */
TEST(test_gridsys, test_niggli_niggli_reduce) {
    /* row vectors */
    const double input_lattice[3][3] = {
        {3.7484850618560537, 0.5179527621527573, 0.2725676000178477},
        {2.5954120365612385, 7.9478524331504028, 0.7065992265067038},
        {3.1731369922769055, 0.0785542643845797, 7.7623356251774149}};
    /* row vectors */
    const double ref_niggli_lattice[3][3] = {
        {-3.7484850618560537, -0.5179527621527573, -0.2725676000178477},
        {-0.5753480695791482, -0.4393984977681776, 7.4897680251595675},
        {-1.1530730252948151, 7.4298996709976457, 0.4340316264888561}};
    double niggli_lattice[9];
    long i, j;
    int succeeded;

    /* row vectors -> column vectors */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            niggli_lattice[i * 3 + j] = input_lattice[j][i];
        }
    }
    succeeded = niggli_reduce(niggli_lattice, 1e-5);
    ASSERT_TRUE(succeeded);
    /* compare row vectors and column vectors */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            ASSERT_LT(
                (fabs(ref_niggli_lattice[j][i] - niggli_lattice[i * 3 + j])),
                1e-5);
        }
    }
}
