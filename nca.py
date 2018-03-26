#!/usr/bin/env python
import sys
from optparse import OptionParser

from tpm import TPM, distance, convert_to_hex


LEARNING_RULES = {
    'hebbian': TPM.learning_rule_hebbian,
    'anti_hebbian': TPM.learning_rule_anti_hebbian,
    'random_walk': TPM.learning_rule_random_walk,
    'custom': TPM.learning_rule_custom,
}


def main(K, N, L, seed, learning_rule):
    a = TPM(K=K, N=N, L=L, learning_rule=learning_rule)
    b = TPM(K=K, N=N, L=L, learning_rule=learning_rule)
    if seed:
        try:
            offset = len(seed)//2
            a.load_from_string(seed[:offset])
            b.load_from_string(seed[offset:])
        except Exception:
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
        rnd_input = a.rnd_input_vector()
        a_out, a_sigma = a.out(rnd_input)
        b_out, b_sigma = b.out(rnd_input)

        if a_out == b_out:
            a.back_propagate(rnd_input, a_out, b_out, a_sigma)
            b.back_propagate(rnd_input, b_out, a_out, b_sigma)

        sys.stdout.write("\rcycles: %010.i; distance: %010.i;" % (
            i, distance(a, b)))
        sys.stdout.flush()
        i += 1

    if i > 1:
        print()
    print("complete")
    print("secret key:", a.hexdigest())


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-K", "--K", dest="K", type="int", default=3,
                      help="hidden neurons")
    parser.add_option("-N", "--N", dest="N", type="int", default=3,
                      help="input neurons by hidden neuron")
    parser.add_option("-L", "--L", dest="L", type="int", default=3,
                      help="radius of possible values of a neuron weights")
    parser.add_option("-s", "--seed", dest="seed",
                      help="starting weights of two networks")
    parser.add_option("-r", "--rule", dest="rule", type="choice",
                      default='hebbian',
                      choices=list(LEARNING_RULES),
                      help='choose from {}'.format(', '.join(LEARNING_RULES)))

    options, args = parser.parse_args()

    try:
        main(options.K, options.N, options.L, options.seed,
             LEARNING_RULES.get(options.rule))
    except KeyboardInterrupt:
        print("\nexit")
