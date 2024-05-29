import argparse
import json
import logging
import os

import pandas as pd

import seleCte.celeskeleton.celeskeleton as celeskeleton
import seleCte.utils as utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_room_batch(
    n_rooms,
    path_folder_to_save,
    test_name,
    d_proba_estimation,
    room_size,
    backtracking_depth,
    verbose=False,
):
    if not os.path.exists(path_folder_to_save):
        os.mkdir(path_folder_to_save)
    for i in range(n_rooms):
        room_data_temp = utils.generate_room_data(
            d_proba_estimation, room_size, backtracking_depth, verbose
        )

        pd.DataFrame(room_data_temp).to_csv(
            f"{path_folder_to_save}/room_{test_name}_{i}.csv",
            header=None,
            index=None,
            sep=";",
        )
        logger.info(
            f"Successfully generated room {i}: saved to {path_folder_to_save}/room_test_{test_name}_{i}.csv"
        )


def main(args):

    nb_rooms = args.nb_rooms
    xp_type = args.experiment_type
    rs = args.room_size
    p = args.proba
    btd = args.bt_depth
    fn = args.folder_name

    with open(
        f"./seleCte/pcg/preprocessing/probability_estimation_dicts/{args.dict_proba_estimation}.json"
    ) as f:
        d_pe = json.load(f)
    f.close()

    if xp_type == "rooms":
        generate_room_batch(
            nb_rooms,
            f"./seleCte/pcg/experiments/{fn}",
            xp_type,
            d_pe,
            rs,
            btd,
            args.verbose,
        )

    else:
        lvl_skeleton = celeskeleton.PCG_skeleton(nb_rooms, p, rs)
        for n_room in range(1, nb_rooms + 1):
            lvl_skeleton.lvl[f"room_{n_room}"].data = utils.generate_room_data(
                d_pe, rs, btd
            )
            lvl_skeleton.lvl[f"room_{n_room}"].create_exits_in_matrix()
            lvl_skeleton.lvl[f"room_{n_room}"].add_respawn_points()
        lvl_skeleton.save(f"../experiments/{fn}")

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb-rooms", "-nr", required=True, type=int, help="Number of rooms to generate"
    )

    parser.add_argument(
        "--experiment-type",
        "-et",
        required=True,
        choices=["rooms", "skel"],
        type=str,
        help="Rooms only or skeleton integration",
    )

    parser.add_argument(
        "--room-size",
        "-rs",
        nargs="+",
        required=False,
        type=int,
        help="Room size as two separate numbers, width then height",
        default=[40, 23],
    )

    parser.add_argument(
        "--bt-depth",
        "-btd",
        required=False,
        type=int,
        help="Depth of backtracking used in the room generation function",
        default=0,
    )

    parser.add_argument(
        "--proba",
        "-p",
        required=False,
        type=float,
        help="Labyrinth proba: 0 for a pathway level, 1 for a completely random order",
        default=0.2,
    )

    parser.add_argument(
        "--folder-name",
        "-fn",
        required=False,
        type=str,
        help="Name of the folder where generated files will be stored",
        default="default_generated_test_rooms",
    )

    parser.add_argument(
        "--dict-proba-estimation",
        "-dpe",
        required=False,
        type=str,
        help="Name of the PE dictionary to use",
        default="pe_d5_full",
    )

    parser.add_argument(
        "--verbose", "-v", required=False, type=bool, help="Verbose", default=False
    )

    args = parser.parse_args()

    main(args)
