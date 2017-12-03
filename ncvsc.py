#!/usr/bin/env python

from functools import reduce
from operator import mul
import sys
from optparse import OptionParser

from tpm import TPM, distance


def main(K, N, L, seed, learning_rule):
    a = TPM(K=K, N=N, L=L, learning_rule=learning_rule)
    b = TPM(K=K, N=N, L=L, learning_rule=learning_rule)
    if seed:
        try:
            offset = len(seed)//2
            a.load_from_string(seed[:offset])
            b.load_from_string(seed[offset:])
        except Exception as e:
            raise e
            print("Bad seed")
            return
    else:
        a.randomize_weights()
        b.randomize_weights()

    print('seed: {}{}'.format(a, b))
    print('rule:', a.learning_rule.__name__[14:])
    print('K:', K)
    print('N:', N)
    print('L:', L)
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

    if i > 1:
        print()
    print("complete")


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-K", "--K", dest="K", type="int", default=3)
    parser.add_option("-N", "--N", dest="N", type="int", default=3)
    parser.add_option("-L", "--L", dest="L", type="int", default=3)
    parser.add_option("-s", "--seed", dest="seed")
    parser.add_option("-r", "--rule", dest="rule")

    options, args = parser.parse_args()

    main(options.K, options.N, options.L, options.seed, options.rule)
