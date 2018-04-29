'''
Created on Nov, 2017

@author: FrancesZhou
'''

from __future__ import absolute_import

import json
import cPickle as pickle


def dump_json(data, file):
    try:
        with open(file, 'w') as datafile:
            json.dump(data, datafile)
    except Exception as e:
        raise e

def load_json(file):
    try:
        with open(file, 'r') as datafile:
            data = json.load(datafile)
    except Exception as e:
        raise e
    return data

def dump_pickle(data, file):
    try:
        with open(file, 'w') as datafile:
            pickle.dump(data, datafile)
    except Exception as e:
        raise e

def load_pickle(file):
    try:
        with open(file, 'r') as datafile:
            data = pickle.load(datafile)
    except Exception as e:
        raise e
    return data

def write_file(data, file):
    try:
        with open(file, 'w') as datafile:
            for line in data:
                datafile.write(str(line[0]) + '\t' + str(line[1]) + '\n')
    except Exception as e:
        raise e

def load_txt(file):
    try:
        with open(file, 'r') as df:
            data = df.readlines()
    except Exception as e:
        raise e
    return data