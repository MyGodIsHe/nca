from hashlib import md5
from array import array
from random import randint, choice


def default_hash_method(weights):
    return md5(weights.tobytes()).digest()


def sgn(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def theta(a, b):
    return -1 if a != b else 1


def distance(a, b):
    cnt = 0
    ln = len(a.weights)
    for i in range(ln):
        cnt += abs(a.weights[i] - b.weights[i])
    return cnt


class TPM(object):

    def __init__(self, K, N, L, hash_method=default_hash_method, learning_rule=''):
        assert isinstance(K, int) and K > 0
        assert isinstance(N, int) and N > 0
        assert isinstance(L, int) and L > 0
        assert hash_method is not None

        self.K = K
        self.N = N
        self.L = L
        self.hash_method = hash_method
        self.learning_rule = {
            'hebbian': self.learning_rule_hebbian,
            'anti_hebbian': self.learning_rule_anti_hebbian,
            'random_walk': self.learning_rule_random_walk,
            'custom': self.learning_rule_custom,
        }.get(learning_rule, self.learning_rule_hebbian)

        self.weights = array('i', range(K*N))

    def __repr__(self):
        return ''.join('{:02x}'.format(x) for x in self.weights.tobytes())

    def load_from_string(self, hex):
        bytes = []
        for i in range(0, len(hex)//2, 2):
            bytes.append(int(hex[i:i + 2], 16))
        for i in range(len(self.weights)):
            self.weights[i] = bytes[i]

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
                self.learning_rule(ij, sigma, input, self_out, him_out)

    def learning_rule_hebbian(self, ij, sigma, input, self_out, him_out):
        self.weights[ij] = self.g(
            self.weights[ij] + sigma * input[ij] * theta(self_out, sigma) * theta(self_out, him_out))

    def learning_rule_anti_hebbian(self, ij, sigma, input, self_out, him_out):
        self.weights[ij] = self.g(
            self.weights[ij] - sigma * input[ij] * theta(self_out, sigma) * theta(self_out, him_out))

    def learning_rule_random_walk(self, ij, sigma, input, self_out, him_out):
        self.weights[ij] = self.g(self.weights[ij] + input[ij] * theta(self_out, sigma) * theta(self_out, him_out))

    def learning_rule_custom(self, ij, sigma, input, self_out, him_out):
        self.weights[ij] = self.g(self.weights[ij] - input[ij] * theta(self_out, sigma) * theta(sigma, him_out))
