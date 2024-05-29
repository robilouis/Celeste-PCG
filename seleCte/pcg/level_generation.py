import argparse
import json
import logging

import pandas as pd

import seleCte.celeskeleton.celeskeleton as celeskeleton
import seleCte.utils as utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_ROOM_SIZE = [40, 23]


def generate_temp_room(d_exits, d_proba_estimation, btd, rs, origin, status):
    temp_room = celeskeleton.Room(width=rs[0], height=rs[1], status=status)
    temp_data = utils.generate_room_data(d_proba_estimation, backtracking_depth=btd)
    temp_room.set_data(temp_data)
    temp_room.set_origin(origin[0], origin[1])
    temp_room.set_exits(d_exits)
    temp_room.create_exits_in_matrix()
    temp_room.add_respawn_points()
    return temp_room


def main(args):

    with open(
        "./seleCte/pcg/preprocessing/probability_estimation_dicts/pe_d5_full.json"
    ) as f:
        d_proba_estimation = json.load(f)
    f.close()

    if args.room_size:
        rs = args.room_size
    else:
        rs = DEFAULT_ROOM_SIZE

    if args.bt_depth:
        btd = args.bt_depth
    else:
        btd = 0

    if args.level_name:
        lvl_name = args.level_name
    else:
        lvl_name = "default_generated_level"

    if args.tries_limit:
        limit = args.tries_limit
    else:
        limit = 10

    if args.tries_limit:
        reset_skeleton = args.reset_skeleton
    else:
        reset_skeleton = False

    nb_rooms = args.nb_rooms
    l_status = ["start"] + [None] * (nb_rooms - 2) + ["end"]

    success_gen = [False for _ in range(nb_rooms)]

    while sum(success_gen) < nb_rooms:

        logger.info("Generating new level skeleton...")
        lvl_skeleton = celeskeleton.PCG_skeleton(nb_rooms, args.proba, rs)

        for room_nb in range(1, nb_rooms + 1):
            is_playable = False
            nb_tries = 0

            if room_nb > 1:
                if not success_gen[
                    room_nb - 2
                ]:  # playability check already failed - no need to go further
                    break

            while not is_playable and nb_tries < limit:
                nb_tries += 1

                logger.info(f"Room {room_nb}, try nb {nb_tries}")

                # Get the exits defined by the skeleton to set those to the generated room
                d_exits_current_room = lvl_skeleton.get_room_by_name(
                    f"room_{room_nb}"
                ).get_exits()
                origin_current_room = lvl_skeleton.get_room_by_name(
                    f"room_{room_nb}"
                ).get_origin()

                # First add the special point if start or end - if it does not work within limit, re-gen room
                sp_point_added = False
                while not sp_point_added:
                    temp_room = generate_temp_room(
                        d_exits_current_room,
                        d_proba_estimation,
                        btd,
                        rs,
                        origin_current_room,
                        l_status[room_nb - 1],
                    )
                    sp_point_added = temp_room.add_special_points(
                        nb_tries_limit=10, verbose=True
                    )

                # At this point, we just have to assert playability of a room - if re-gen the room data is not enough
                # it means that the exits are probably messed up: re-gen the skeleton
                is_playable = temp_room.is_playable_room(verbose=True)
                if is_playable:
                    success_gen[room_nb - 1] = True
                    logger.info(f"Generated room {room_nb} in {nb_tries} iterations.")

                lvl_skeleton.get_room_by_name(f"room_{room_nb}").set_data(
                    temp_room.data
                )

            # Got out of the loop: either room is playable with a special point, or something is likely wrong with the
            # exits placement => re-gen another skeleton
            if not is_playable:  # reset success_gen and break out the for loop
                if reset_skeleton:
                    success_gen = [False for _ in range(nb_rooms)]
                    logger.warning(
                        f"Room {room_nb} generation failed. Re-generating a skeleton.\n"
                    )
                    break
                else:
                    success_gen[room_nb - 1] = True
                    logger.warning(
                        f"Room {room_nb} generation failed. Pursuing generation anyways.\n"
                    )

    lvl_skeleton.format_filled_celeskeleton()
    lvl_skeleton.save(f"{lvl_name}")

    logger.info(
        f"Celeskeleton object has been correcly formatted and saved in folder {lvl_name}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nb-rooms", "-nr", required=True, type=int, help="Number of rooms to generate"
    )
    parser.add_argument(
        "--proba",
        "-p",
        required=True,
        type=float,
        help="Labyrinth proba: 0 for a pathway level, 1 for a completely random order",
    )
    parser.add_argument(
        "--room-size",
        "-rs",
        nargs="+",
        required=False,
        type=int,
        help="Room size as two separate numbers, width then height",
    )
    parser.add_argument(
        "--bt-depth",
        "-btd",
        required=False,
        type=int,
        help="Depth of backtracking used in the room generation function",
    )
    parser.add_argument(
        "--level-name",
        "-ln",
        required=False,
        type=str,
        help="Name of the folder where generated files will be stored",
    )
    parser.add_argument(
        "--tries-limit",
        "-tl",
        required=False,
        type=int,
        help="Limit of tries per room gen",
    )
    parser.add_argument(
        "--reset-skeleton",
        "-r",
        required=False,
        type=bool,
        help="Resetting skeleton if room gen fails",
    )
    args = parser.parse_args()

    main(args)
