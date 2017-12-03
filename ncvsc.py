#!/usr/bin/env python

from functools import reduce
from operator import mul
import sys

from tpm import TPM, distance


if __name__ == '__main__':
    K = 3
    N = 3
    L = 3
    a = TPM(K=K, N=N, L=L)
    a.randomize_weights()
    b = TPM(K=K, N=N, L=L)
    b.randomize_weights()

    print("start distance: ", distance(a, b))
    i = 1

    while a.hash_of_weights() != b.hash_of_weights():
        a_input = a.rnd_input_vector()
        a_sigma = list(a.propagate(a_input))
        a_out = reduce(mul, a_sigma)

        b_input = b.rnd_input_vector()
        b_sigma = list(b.propagate(b_input))
        b_out = reduce(mul, b_sigma)

        if a_out == b_out:
            a.back_propagate(a_input, a_out, b_out, a_sigma)
            b.back_propagate(b_input, b_out, a_out, b_sigma)

        sys.stdout.write("\rcycles: %010.i; distance: %010.i;" % (i, distance(a, b)))
        sys.stdout.flush()
        i += 1
