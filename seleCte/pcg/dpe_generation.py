import argparse
import itertools
import json
import os
import sys

sys.path.append("..")

import numpy as np
import pandas as pd

from collections import Counter

from seleCte.utils import (
    ALL_TILES_AND_ENTITIES,
    submatrix_to_count,
    get_all_submatrices,
)

DATA_PATH = "./pcg/preprocessing/data_pcg_ready/all_rooms/"


def main(args):

    n_mat = int(np.sqrt(len(args.mdmc_matrix)))
    mdmc_matrix = np.fromstring(",".join(args.mdmc_matrix), dtype=int, sep=",").reshape(
        n_mat, n_mat
    )

    d_absolute_counts = {}
    for perm in itertools.product(
        ALL_TILES_AND_ENTITIES, repeat=Counter(mdmc_matrix.flatten())[1]
    ):
        d_absolute_counts["".join(perm)] = []

    for fn in os.listdir(DATA_PATH):
        if fn[0] in args.dataset:
            df_temp = pd.read_csv(
                os.path.join(DATA_PATH, fn), header=None, sep=";", dtype=str
            ).map(lambda x: "0" if x == "" or x == " " else x)
            lvl_temp = df_temp.to_numpy(dtype=str)

            for submatrix in get_all_submatrices(lvl_temp):
                pattern, tiletype = submatrix_to_count(submatrix, mdmc_matrix)
                d_absolute_counts[pattern].append(tiletype)

    for key in d_absolute_counts.keys():
        d_absolute_counts[key] = dict(Counter(d_absolute_counts[key]))

    d_proba_estimation = {}

    for key in d_absolute_counts.keys():
        d_temp = d_absolute_counts[key]
        if d_temp:
            d_proba_estimation[key] = {
                k: (v / total)
                for total in (sum(d_temp.values()),)
                for k, v in d_temp.items()
            }

    for pattern in d_proba_estimation.keys():
        for symbol in ALL_TILES_AND_ENTITIES:
            if symbol not in d_proba_estimation[pattern].keys():
                d_proba_estimation[pattern][symbol] = 0.0

    fn_result_dpe = f"./pcg/preprocessing/probability_estimation_dicts/{args.dataset}_{args.mdmc_matrix}.json"
    with open(fn_result_dpe, "w") as file:
        json.dump(d_proba_estimation, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mdmc-matrix",
        "-mm",
        required=False,
        type=str,
        help="MdMC matrix chosen for DPE generation",
        default="000011012",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=False,
        type=str,
        help="Number of levels considered for training (9 for LostLevels)",
        default="0123456789L",
    )
    args = parser.parse_args()

    main(args)
