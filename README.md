# Neural Cryptographic Algorithm

The most used protocol for key exchange between two parties A and B in the practice is Diffie-Hellman protocol. Neural key exchange, which is based on the synchronization of two tree parity machines, should be a secure replacement for this method. Synchronizing these two machines is similar to synchronizing two chaotic oscillators in chaos communications.

## Usage
```
nca.py [options]

Options:
  -h, --help            show this help message and exit
  -K K, --K=K           hidden neurons
  -N N, --N=N           input neurons by hidden neuron
  -L L, --L=L           radius of possible values of a neuron weights
  -s SEED, --seed=SEED  starting weights of two networks
  -r RULE, --rule=RULE  choose from hebbian, anti_hebbian, random_walk, custom
```

## Example
```
seed: 01000000ffffffffffffffff00000000ffffffff0200000001000000fdffffffffffffffffffffffffffffff020000000300000002000000000000000200000002000000feffffff
rule: hebbian
K: 3
N: 3
L: 3
start distance:  20
cycles: 0000000172; distance: 0000000000;
complete
secret key: 21DE32F

```