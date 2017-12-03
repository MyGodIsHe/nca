from hashlib import md5
from array import array
from random import randint


def default_hash_method(weights):
    return md5(weights.tobytes()).digest()


def sgn(x):
    return 1 if x > 0 else -1


def cmp(a, b):
    return 1 if a == b else -1


class TPM(object):

    def __init__(self, K, N, L, hash_method=default_hash_method):
        assert isinstance(K, int) and K > 0
        assert isinstance(N, int) and N > 0
        assert isinstance(L, int) and L > 0
        assert hash_method is not None

        self.K = K
        self.N = N
        self.L = L
        self.hash_method = hash_method

        self.weights = array('f', range(K*N))

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
            1 if randint(0, 1) == 0 else -1
            for i in range(len(self.weights))
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

    def back_propagate(self, input, self_out, him_out, sigma_list):
        for i in range(self.K):
            sigma = sigma_list[i]
            for j in range(self.N):
                ij = i * self.N + j
                self.weights[ij] = self.g(self.weights[ij] - input[ij] * cmp(self_out, sigma) * cmp(sigma, him_out))
