from __future__ import print_function
import numpy as np
import scipy.stats
import os
import argparse


def truncnorm(lower, upper, mean, width):
    return scipy.stats.truncnorm((lower - mean) / width,
                                 (upper - mean) / width,
                                 loc=mean, scale=width).pdf


class Species(object):

    def __init__(self, name, t0, pref=None, sig=None):
        # Label for species, usually just a unique integer in practice
        self.name = name

        # Trait to which species is best suited
        if pref is not None:
            self.pref = pref
        else:
            self.pref = np.random.uniform(0.0, 1.0)

        # Degree of generalisation
        if sig is not None:
            self.sig = sig
        else:
            self.sig = 10.0 ** np.random.uniform(-3.0, 3.0)

        # Time at which species first arrived into environment
        self.t0 = t0

        # Gaussian curve representing how species responds to a location
        # with a particular trait
        self.response = truncnorm(0.0, 1.0, self.pref, self.sig)

    def __repr__(self):
        return 'Name: {}, pref: {}, sig: {}, t0: {}'.format(
            self.name, self.pref, self.sig, self.t0)

    __str__ = __repr__

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['response']
        return d


def initialise_env(size, species_list, hetero, trait_0=None):
    env = {}

    # Species
    env['s'] = np.random.choice(species_list, size=size)

    # Traits
    if hetero:
        env['t'] = np.random.uniform(0.0, 1.0, size=size)
    else:
        trait_0 = 0.5
        env['t'] = np.ones([size], dtype=np.float) * trait_0
    return env


def initialise_species_list(n):
    return [(Species(i, 0)) for i in range(n)]


def eco(m, n, t_max, pn=0.0, hetero=False, out=None, seed=None):
    '''
    Simulate an ecosystem environment with a number of locations, and a number
    of species who are differently adapted to locations within the
    environment. Processes include expansion of well-suited species,
    and immigration of new species.
    'm': Number of locations
    'n': Number of species initially in the environment
    't_max': Number of time-steps to run the simulation for
    'pn': Probability per location to introduce a
        new species
    'hetero': Whether or not the environment varies,
        or whether all locations are identical
    'out': Directory in which to dump output data
    'seed': Random number generator seed
    '''

    # Set random number generator seed
    np.random.seed(seed)

    # Initialise time
    t = 0

    species_list = initialise_species_list(n)
    env = initialise_env(m, species_list, hetero)

    # Output static system data
    if out is not None:
        if not os.path.isdir(out):
            os.makedirs(os.path.join(out, 'dyn'))
        np.savez(os.path.join(out, 'static.npz'), traits=env['t'])

    # Start main simulation loop
    while t < t_max:
        # How suitable the species in each location are to be in all other
        # locations
        suits = np.array([s.response(env['t']) for s in env['s']])
        # Normalise to form probability distribution
        suits /= suits.sum(axis=0)

        # Array to hold new environment species state
        s_new = np.empty([m], dtype=Species)

        # For each location
        for i in range(m):
            # Either introduce new species
            if np.random.uniform() < pn:
                # Make a new species with a new unique name
                species_new = Species(len(species_list) + 1, t)
                # Add to the master species list
                species_list.append(s_new)
            # Otherwise species expands from within the system
            else:
                i_species_new = np.random.choice(m, p=suits[:, i])
                species_new = env['s'][i_species_new]

            s_new[i] = species_new

        env['s'] = s_new

        if out is not None:
            fname = os.path.join(out, 'dyn', '{:010d}'.format(t))
            np.savez(fname, species=env['s'], t=t)

        print(t)

        t += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run well-mixed ecosystem simulation')
    parser.add_argument('-o', '--out',
                        help='Directory in which to output data')
    parser.add_argument('-m', type=int,
                        help='Number of locations')
    parser.add_argument('-n', '--n', type=int,
                        help='Initial number of species')
    parser.add_argument('-t', '--t_max', type=int,
                        help='Number of time-steps to run for')
    parser.add_argument('--hetero', default=False, action='store_true',
                        help='Make hetero environment')
    parser.add_argument('-pn', type=float, default=0.0,
                        help='Probability of introduction of new species per '
                             'location')
    parser.add_argument('-s', '--seed', type=int,
                        help='Random number generator seed')
    args = parser.parse_args()

    eco(m=args.m, n=args.n, t_max=args.t_max, pn=args.pn, out=args.out,
        hetero=args.hetero, seed=args.seed)
