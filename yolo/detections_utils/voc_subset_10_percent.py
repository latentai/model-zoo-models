import os

path_to_subset = os.path.join(os.path.dirname(__file__), 'voc_subset_10_percent.txt')

with open(path_to_subset) as fp:
    voc_subset = fp.readlines()
