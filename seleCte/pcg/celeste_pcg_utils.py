import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


ALL_TILES_AND_ENTITIES = [
    "0",
    "1",
    "^",
    "<",
    "v",
    ">",
    "D",
    "C",
    "_",
    "O",
    "Q",
    "W",
    "S",
    "B",
    "R",
    "F",
    "L",
]


def color_map(val):
    color_dict = {
        "1": "salmon",
        "<": "grey",
        ">": "grey",
        "v": "grey",
        "^": "grey",
        "P": "pink",
        "W": "blue",
        "D": "purple",
        "C": "grey",
        "_": "brown",
        "O": "green",
        "Q": "red",
        "S": "orange",
        "B": "cyan",
        "R": "lime",
        "F": "yellow",
        "L": "lightblue",
        "0": None,
    }
    return "background-color: %s" % color_dict[val]


def visualize_room(array):
    df_visu = pd.DataFrame(array)
    df_visu = df_visu.style.map(color_map)
    return df_visu


def generate_empty_room(width=40, height=23):
    room = np.zeros((height, width), dtype=str)
    room[0, :] = "1"
    room[:, 0] = "1"
    room[-1, :] = "1"
    room[:, -1] = "1"
    room[1, 1:-1] = "0"
    room[1:-1, 1] = "0"

    return room


def extract_pattern(array, x, y):
    return "".join((array[x + 1, y + 1], array[x + 1, y + 2], array[x + 2, y + 1]))


def extract_proba_from_tile(array, x, y, d_pe):
    pattern = extract_pattern(array, x, y)

    try:
        proba_dist = d_pe[pattern]
    except KeyError:  # unseen state
        proba_dist = {}

    return proba_dist


def generate_new_tile(proba_dist, all_symbols=ALL_TILES_AND_ENTITIES, exclude=[]):
    if not proba_dist:  # unseen state
        new_tile = np.random.choice(all_symbols)
    else:
        if [key for key in proba_dist.keys() if proba_dist[key] > 0] == exclude:
            raise AssertionError(
                "There is no symbol with a strictly positive proba left!"
            )
        new_tile = np.random.choice(
            a=list(proba_dist.keys()), p=list(proba_dist.values())
        )
        while new_tile in exclude:
            new_tile = np.random.choice(
                a=list(proba_dist.keys()), p=list(proba_dist.values())
            )

    return new_tile


def update_room_array(array, x, y, d_proba_estimation, bt_depth=0, verbose=True):

    bt_depth_true = bt_depth_true = min(bt_depth, array.shape[1] - y - 3)
    # if bt depth > 0, needs to apply backtracking
    # everytime you pick a symbol, add it to excluded - if cannot pick: random pick
    if bt_depth_true:
        excluded_symbols = {i: [] for i in range(bt_depth_true)}
        k = 0
        array_temp = array.copy()

        while k >= 0 and k < bt_depth_true:
            try:
                pb_dist = extract_proba_from_tile(
                    array_temp, x, y + k, d_pe=d_proba_estimation
                )

                if pb_dist:
                    nt = generate_new_tile(pb_dist, exclude=excluded_symbols[k])
                    excluded_symbols[k].append(nt)
                    array_temp[x + 2, y + 2 + k] = nt
                    k += 1
                else:  # if pb_dist empty: unseen state
                    k -= 1

            except (
                AssertionError
            ):  # no tiles left for level k - need to try something else at k-1
                k -= 1  # backtracking

        if k == -1:  # no matter the tile used in base lvl it fails => random gen
            logger.warning("Backtracking failed! Random generation.")
            array[x + 2, y + 2] = generate_new_tile({})
        else:  # we made it through: tile at the top level works fine
            array[x + 2, y + 2] = array_temp[x + 2, y + 2]

    else:
        array[x + 2, y + 2] = generate_new_tile(
            extract_proba_from_tile(array, x, y, d_pe=d_proba_estimation)
        )

    if verbose:
        logger.info(f"New tile generated in position ({x+2}, {y+2}): {array[x+2, y+2]}")

    return array


def generate_room(
    d_proba_estimation, width=40, height=23, backtracking_depth=0, verbose=False
):
    room = generate_empty_room(width, height)

    for x in range(height - 2):
        for y in range(width - 2):
            room = update_room_array(
                room,
                x,
                y,
                d_proba_estimation,
                bt_depth=backtracking_depth,
                verbose=verbose,
            )

    return room


def generate_room_batch(
    n_rooms,
    path_folder_to_save,
    d_proba_estimation,
    width=40,
    height=23,
    backtracking_depth=0,
    verbose=False,
):
    if not os.path.exists(path_folder_to_save):
        os.mkdir(path_folder_to_save)
    for i in range(n_rooms):
        room_temp = generate_room(
            d_proba_estimation, width, height, backtracking_depth, verbose
        )
        pd.DataFrame(room_temp).to_csv(
            f"{path_folder_to_save}/room_{i}_generated_MdMC.csv",
            header=None,
            index=None,
            sep=";",
        )
        logger.info(
            f"Successfully generated room {i}: saved to {path_folder_to_save}/room_{i}_generated_MdMC_{width}_{height}.csv"
        )
