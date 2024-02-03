import argparse
import json
import logging

import seleCte.celeskeleton.celeskeleton as celeskeleton
import seleCte.pcg.celeste_pcg_utils as utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_ROOM_SIZE = [40, 23]



def main(args):
    
    with open("./seleCte/pcg/preprocessing/probability_estimation.json", "r") as f:
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

    pcg_gen_lvl = celeskeleton.PCG_skeleton(args.nb_rooms, args.proba, rs)

    for room_nb in range(1, args.nb_rooms+1):
        temp_data = utils.generate_room(d_proba_estimation, backtracking_depth=btd)
        pcg_gen_lvl.get_room_by_name(f"room_{room_nb}").set_data(temp_data)

    celeskeleton.format_filled_celeskeleton(pcg_gen_lvl)

    pcg_gen_lvl.save("test_model_py")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb-rooms", "-nr", required=True, type=int, help="Number of rooms to generate")
    parser.add_argument("--proba", "-p", required=True, type=float, help="Labyrinth proba: 0 for a pathway level, 1 for a completely random order")
    parser.add_argument("--room-size", "-rs", nargs="+", required=False, type=int, help="Room size as two separate numbers, width then height")
    parser.add_argument("--bt-depth", "-btd", required=False, type=int, help="Depth of backtracking used in the room generation function")
    args = parser.parse_args()

    main(args)
