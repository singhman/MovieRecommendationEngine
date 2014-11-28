import numpy as np
from math import ceil

class SplitSampling(object):
    """ Random Split Sampling the dataset into two sets.
    """
    def __init__(self, n, evaluation_fraction=0.7, indices=False,
            random_state=None):
        self.n = n
        self.evaluation_fraction = evaluation_fraction
        self.random_state = random_state
        self.indices = indices

    def split(self, evaluation_fraction=None, indices=False, random_state=None, permutation=True):
        if evaluation_fraction is not None:
            self.evaluation_fraction = evaluation_fraction
        if random_state is not None:
            self.random_state = random_state

        self.indices = indices

        rng = self.random_state = check_random_state(self.random_state)
        n_train = ceil(self.evaluation_fraction * self.n)
        #random partition
        permutation = rng.permutation(self.n) if permutation \
                             else np.arange(self.n)
        ind_train = permutation[-n_train:]
        ind_ignore = permutation[:-n_train]
        if self.indices:
            return ind_train, ind_ignore
        else:
            train_mask = np.zeros(self.n, dtype=np.bool)
            train_mask[ind_train] = True
            test_mask = np.zeros(self.n, dtype=np.bool)
            test_mask[ind_ignore] = True
            return train_mask, test_mask

    def __repr__(self):
        return ('%s(%d, evaluation_fraction=%s, indices=%s, '
                'random_state=%d)' % (
                    self.__class__.__name__,
                    self.n,
                    str(self.evaluation_fraction),
                    self.indices,
                    self.random_state,
                ))

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
