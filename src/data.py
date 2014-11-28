from os.path import dirname
from os.path import join
import numpy as np

class Data(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def load_dataset(*args):
    if args:
        if len(args) == 2:
            user_file = args[0]
            item_file = args[1]
        else:
            print 'Please provide both user and item file'
    else:
        user_file = 'u.data'
        item_file = 'u.item'
    base_dir = join(dirname(__file__), 'data/')
    # Read data
    data_m = np.loadtxt(base_dir + user_file, delimiter='\t', usecols=(0, 1, 2), dtype=int)
    data_movies = {}
    for user_id, item_id, rating in data_m:
        data_movies.setdefault(user_id, {})
        data_movies[user_id][item_id] = int(rating)

    #Read the titles
    data_titles = np.loadtxt(base_dir + item_file, delimiter='|', usecols=(0, 1), dtype=str)

    data_t = []
    for item_id, label in data_titles:
        data_t.append((int(item_id), label))
    data_titles = dict(data_t)
    fdescr = open(base_dir + 'README')
    return Data(data=data_movies, item_ids=data_titles,user_ids=None, DESCR=fdescr.read())
