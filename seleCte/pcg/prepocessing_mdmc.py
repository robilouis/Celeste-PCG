import itertools
import json
import numpy as np

from collections import Counter


mdmc_matrices = {
    "d1": np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 2],
            ]
        ),
    "d2": np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 2],
            ]
        ),
    "d3": np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 2],
            ]
        ),
    "d4": np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 1, 2],
            ]
        ),
    "d5": np.array(
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 2],
            ]
        ),
    "d6": np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [1, 1, 2],
            ]
        ),
}

def submatrix_to_count(array, mdmc_matrix):
    idx = np.where(mdmc_matrix==1)
    return "".join((array[idx[0][k], idx[1][k]] for k in range(idx[0].shape[0]))), array[2, 2]

def get_all_submatrices(array, xmax=23, ymax=40, size=2):
    all_submatrices = []
    for x in range(xmax-size):
        for y in range(ymax-size):
            all_submatrices.append(array[x:x+3, y:y+3])
    return all_submatrices

mdmc_matrix = mdmc_matrices["d5"]

with open("../data_pcg_ready/40x23_fg/lvls_fg.json", "r") as f:
    d_levels = json.load(f)

all_tiles = []
for lv in d_levels.values():
    all_tiles.extend(np.array(lv).flatten())
all_tiles = set(all_tiles)

d_absolute_counts = {}
for perm in itertools.product(all_tiles, repeat=3):
    d_absolute_counts["".join(perm)] = []

for lvl in d_levels.values():
    for submatrix in get_all_submatrices(np.array(lvl)):
        pattern, tiletype = submatrix_to_count(submatrix, mdmc_matrix)
        d_absolute_counts[pattern].append(tiletype)

for key in d_absolute_counts.keys():
    d_absolute_counts[key] = dict(Counter(d_absolute_counts[key]))

d_proba_estimation = {}

for key in d_absolute_counts.keys():
    d_temp = d_absolute_counts[key]
    if d_temp:        
        d_proba_estimation[key] = {k: (v / total) for total in (sum(d_temp.values()),) for k, v in d_temp.items()}