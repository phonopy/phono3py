#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

long get_num_unique_elems(const long array[], const long array_size) {
    long i, num_unique_elems;
    long *unique_elems;

    num_unique_elems = 0;
    unique_elems = (long *)malloc(sizeof(long) * array_size);
    for (i = 0; i < array_size; i++) {
        unique_elems[i] = 0;
    }
    for (i = 0; i < array_size; i++) {
        unique_elems[array[i]]++;
    }
    for (i = 0; i < array_size; i++) {
        if (unique_elems[i]) {
            num_unique_elems++;
        }
    }
    free(unique_elems);
    unique_elems = NULL;
    return num_unique_elems;
}
