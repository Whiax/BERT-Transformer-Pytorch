# -*- coding: utf-8 -*-

import os
from os.path import join
import pickle
import matplotlib.pyplot as plt



def cache_object(obj, name, folder="cache"):
    pickle.dump(obj, open(join(folder, name+'.pickle'), 'wb'), pickle.HIGHEST_PROTOCOL)
    
def load_object(name, folder='cache'):
    return pickle.load(open(join(folder, name + '.pickle'), 'rb'))

def is_object_cached(name, folder='cache'):
    return os.path.exists(join(folder, name + '.pickle'))

#sort a dictionary
def sort_dict(d, i='values', m='desc'):
    if type(i) is str:
        i = 0 if i == 'keys' else 1
    if type(m) is str:
        m = 1 if m == 'asc' else -1
    return {k:v for k,v in sorted(d.items(), key=lambda item: m*item[i])}

#dict to plot
def plot_dict(d, l='', source=None, **kwargs):
    if source is None: source = plt
    d = sort_dict(d,0,1)
    if l != '':
        source.plot(d.keys(), d.values(), label=l, **kwargs)
        source.legend(loc="upper left")
    else:
        source.plot(d.keys(), d.values(), **kwargs)

