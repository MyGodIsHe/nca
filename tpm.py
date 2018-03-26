from hashlib import md5
from array import array
from random import randint, choice
from functools import reduce
from operator import mul


def default_hash_method(weights):
    return md5(weights.tobytes()).digest()


def convert_to_hex(number, base):
    value = sum(
        base ** n * v
        for n, v in enumerate(number[::-1])
    )
    return hex(int(bin(value), 2))[2:].upper()


def sgn(x):
    if x > 0:
        return 1
    if x <= 0:
        return -1


def theta(a, b):
    return 0 if a != b else 1


def distance(a, b):
    cnt = 0
    ln = len(a.weights)
    for i in range(ln):
        cnt += abs(a.weights[i] - b.weights[i])
    return cnt


class TPM(object):

    def __init__(self, K, N, L,
                 learning_rule,
                 hash_method=default_hash_method):
        assert isinstance(K, int) and K > 0
        assert isinstance(N, int) and N > 0
        assert isinstance(L, int) and L > 0
        assert hash_method is not None

        self.K = K
        self.N = N
        self.L = L
        self.hash_method = hash_method
        self.learning_rule = learning_rule

        self.weights = array('i', range(K*N))

    def __repr__(self):
        return ''.join('{:02x}'.format(x) for x in self.weights.tobytes())

    def load_from_string(self, hex):
        bytes = []
        for i in range(0, len(hex)//2, 2):
            bytes.append(int(hex[i:i + 2], 16))
        for i in range(len(self.weights)):
            self.weights[i] = bytes[i]

    def hexdigest(self):
        weights = [w + self.L for w in self.weights]
        return convert_to_hex(weights, 2 * self.L + 1)

    def g(self, x):
        if x > self.L:
            return self.L
        if x < -self.L:
            return -self.L
        return x

    def randomize_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] = randint(-self.L, self.L)

    def rnd_input_vector(self):
        return [
            choice([-1, 0, 1])
            for _ in self.weights
        ]

    def hash_of_weights(self):
        return self.hash_method(self.weights)

    def propagate(self, input_vector):
        for i in range(self.K):
            sigma = 0
            for j in range(self.N):
                ij = i * self.N + j
                sigma += self.weights[ij] * input_vector[ij]
            yield sgn(sigma)

    def out(self, input_data):
        sigma_list = list(self.propagate(input_data))
        value = reduce(mul, sigma_list)
        return value, sigma_list

    def back_propagate(self, input, self_out, him_out, sigma_list):
        for i in range(self.K):
            sigma = sigma_list[i]
            for j in range(self.N):
                ij = i * self.N + j
                self.weights[ij] = self.g(
                    self.weights[ij] +
                    self.learning_rule(sigma, input[ij], self_out, him_out)
                )

    @staticmethod
    def learning_rule_hebbian(sigma, input, self_out, him_out):
        return sigma * input * \
               theta(self_out, sigma) * theta(self_out, him_out)

    @staticmethod
    def learning_rule_anti_hebbian(weight, sigma, input, self_out, him_out):
        return - sigma * input * \
               theta(self_out, sigma) * theta(self_out, him_out)

    @staticmethod
    def learning_rule_random_walk(weight, sigma, input, self_out, him_out):
        return input * theta(self_out, sigma) * theta(self_out, him_out)

    @staticmethod
    def learning_rule_custom(weight, sigma, input, self_out, him_out):
        return - input * theta(self_out, sigma) * theta(sigma, him_out)
