import sys

sys.path.append("../")

import celeskeleton.celeskeleton as celeskeleton
import celeste_pcg_utils as utils

DEFAULT_ROOM_SIZE = [40, 23]

skel_pcg = celeskeleton.PCG_skeleton(10, 0.1, DEFAULT_ROOM_SIZE)

for room_nb in range(1, 11):
    temp_data = utils.generate_room(backtracking_depth=5)
    skel_pcg.get_room_by_name(f"room_{room_nb}").set_data(temp_data)

print("oui ljwhfwljnfw")
