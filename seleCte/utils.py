import heapq
import logging
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import seleCte.celeskeleton.celeskeleton as celeskeleton

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

LETHAL_ENTITIES = ["^", "<", "v", ">", "S"]

NL_ENTITES = [
    "D",
    "C",
    "_",
    "O",
    "Q",
    "W",
    "B",
    "R",
    "F",
]

VALID_FG_TILES = [
    "1",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "G",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "z",
]

VALID_BG_TILES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "a",
    "b",
    "c",
    "d",
    "e",
]

MDMC_MATRICES = {
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
    "d7": np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 1, 2],
        ]
    ),
}


# Functions for DPE extraction
def submatrix_to_count(array, mdmc_matrix):
    idx = np.where(mdmc_matrix == 1)
    return (
        "".join(array[idx[0][k], idx[1][k]] for k in range(idx[0].shape[0])),
        array[-1, -1],
    )


def get_all_submatrices(array, size=2):
    xmax, ymax = array.shape
    all_submatrices = []
    for x in range(xmax - size):
        for y in range(ymax - size):
            all_submatrices.append(array[x : x + size + 1, y : y + size + 1])
    return all_submatrices


# Functions for visualization
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
        "H": "black",
        "0": None,
        "p": "violet",
        "e": "plum",
    }
    return "background-color: %s" % color_dict[val]


def visualize_room(array):
    df_visu = pd.DataFrame(array)
    df_visu = df_visu.style.map(color_map)
    return df_visu


def generate_empty_room_data(width=40, height=23):
    room_data = np.zeros((height, width), dtype=int).astype(str)
    room_data[0, :] = "1"
    room_data[:, 0] = "1"
    room_data[-1, :] = "1"
    room_data[:, -1] = "1"

    return room_data


# Helper functions for extracting pattern proba for MdMC model
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
        if new_tile in exclude:
            new_tile = np.random.choice(
                a=list(proba_dist.keys()), p=list(proba_dist.values())
            )

    return new_tile


def update_room_array(array, x, y, d_proba_estimation, bt_depth=0, verbose=True):
    bt_depth_true = min(bt_depth, array.shape[1] - y - 3)
    # if bt depth > 0, needs to apply backtracking
    # everytime you pick a symbol, add it to excluded - if cannot pick: random pick
    if bt_depth_true:
        excluded_symbols = {i: [] for i in range(bt_depth_true)}
        k = 0
        n_iter_btd = 0
        array_temp = array.copy()

        while k >= 0 and k < bt_depth_true and n_iter_btd < 100:
            n_iter_btd += 1
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

        if n_iter_btd == 100:  # btd got stuck in a back and forth loop
            logger.warning("Backtracking stuck in loop! Random generation.")
            array[x + 2, y + 2] = generate_new_tile({})
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


def generate_room_data(
    d_proba_estimation, room_size, backtracking_depth=0, verbose=False
):
    width, height = room_size[0], room_size[1]
    room_data = generate_empty_room_data(width, height)

    for x in range(height - 2):
        for y in range(width - 2):
            room_data = update_room_array(
                room_data,
                x,
                y,
                d_proba_estimation,
                bt_depth=backtracking_depth,
                verbose=verbose,
            )

    return room_data


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
        room_data_temp = generate_room_data(
            d_proba_estimation, width, height, backtracking_depth, verbose
        )
        pd.DataFrame(room_data_temp).to_csv(
            f"{path_folder_to_save}/room_{i}_generated_MdMC.csv",
            header=None,
            index=None,
            sep=";",
        )
        logger.info(
            f"Successfully generated room {i}: saved to {path_folder_to_save}/room_{i}_generated_MdMC_{width}_{height}.csv"
        )


def generate_temp_room(d_exits, d_proba_estimation, btd, rs, origin, status):
    temp_room = celeskeleton.Room(width=rs[0], height=rs[1], status=status)
    temp_data = generate_room_data(d_proba_estimation, rs, backtracking_depth=btd)
    temp_room.set_data(temp_data)
    temp_room.set_origin(origin[0], origin[1])
    temp_room.set_exits(d_exits)
    temp_room.create_exits_in_matrix()
    temp_room.add_respawn_points()
    return temp_room


# Helper functions for playability module
class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0  # Cost from start to node
        self.h = 0  # Heuristic cost from node to end
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f


def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


def astar(
    room,
    maze,
    start,
    end,
    allow_diagonal_movement=True,
    verbose=False,
    stop_condition=0,
):
    """
    Returns - if exists - a list of tuples as a path from the given start to the given end in the given maze
    """

    nb_iter = 0

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Heapify the open_list and Add the start node
    heapq.heapify(open_list)
    heapq.heappush(open_list, start_node)

    # Adding a stop condition
    outer_iterations = 0
    if stop_condition == 0:
        max_iterations = len(maze[0]) * len(maze)
    else:
        max_iterations = stop_condition

    # what squares do we search
    if allow_diagonal_movement:
        adjacent_squares = (
            (0, -1),  # left
            (0, 1),  # right
            (-1, 0),  # up
            (1, 0),  # down
            (-1, -1),  # up-left
            (-1, 1),  # up-right
            (1, -1),  # down-left
            (1, 1),  # down-right
        )
        direction_cost = (0.1, 0.1, 0.05, 0, 0.5, 0.5, 0.01, 0.01)
        # direction_cost = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        adjacent_square_pick_index = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0))
        direction_cost = (1.0, 1.0, 1.0, 1.0)
        adjacent_square_pick_index = [0, 1, 2, 3]

    # Loop until you find the end
    while len(open_list) > 0:
        nb_iter += 1
        # Randomize the order of the adjacent_squares_pick_index to avoid a decision making bias
        random.shuffle(adjacent_square_pick_index)
        outer_iterations += 1

        if outer_iterations > max_iterations:
            # if we hit this point return the path such as it is
            # it will not contain the destination
            if verbose:
                logger.warning(f"Iteration limit exceeded - stopped at iter {nb_iter}")
            return None

        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            if verbose:
                print(f"Found a path in {nb_iter} iterations")
            return return_path(current_node)

        # Generate children
        children = []

        for pick_index in adjacent_square_pick_index:
            new_position = adjacent_squares[pick_index]
            direction_cost_factor = direction_cost[pick_index]

            # Get node position
            node_position = (
                current_node.position[0] + new_position[0],
                current_node.position[1] + new_position[1],
            )

            # Make sure within range
            if (
                node_position[0] > (len(maze) - 1)
                or node_position[0] < 0
                or node_position[1] > (len(maze[len(maze) - 1]) - 1)
                or node_position[1] < 0
            ):
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if (
                len(
                    [
                        closed_child
                        for closed_child in closed_list
                        if closed_child == child
                    ]
                )
                > 0
            ):
                continue

            # Create the f, g, and h values
            # x, y = child.position
            # local_path = [(x, k) for k in range(y-3, y+4) if 0 <= k < len(maze[0])]
            child.g = current_node.g + (
                direction_cost_factor + get_min_dist_to_nle(room, child.position) / 10
            ) * len(maze[0]) * len(maze)
            # child.g = current_node.g + direction_cost_factor
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                (child.position[1] - end_node.position[1]) ** 2
            )
            child.f = child.g + child.h

            # Child is already in the open list
            if (
                len(
                    [
                        open_node
                        for open_node in open_list
                        if child.position == open_node.position
                        and child.g > open_node.g
                    ]
                )
                > 0
            ):
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    if verbose:
        logger.warning("No path to destination")
    return None


## Evaluation time
def create_paths_heatmap(room, nb_paths, c_map="viridis", max_iter=0):
    l_path = [room.is_playable_room(True, max_iter=max_iter) for _ in range(nb_paths)]

    heatmap_data = np.zeros(room.data.shape, dtype=int)
    for pos in [
        (x, y)
        for x, y in zip(np.where(room.data != "0")[0], np.where(room.data != "0")[1])
    ]:
        heatmap_data[pos] = -1

    for path in l_path:
        if type(path) != bool:
            for path_pos in path:
                heatmap_data[path_pos] += 1

    df = pd.DataFrame(heatmap_data)

    max_value = df.max().max()
    min_value = 0
    norm = plt.Normalize(min_value, max_value)
    cmap = plt.get_cmap(c_map)

    # Step 3: Function to apply the color scale
    def color_scale(val):
        if val == -1:
            color = (0.6, 0.6, 0.6, 1.0)
        elif val == 0:
            color = (0.1, 0.1, 0.1, 1.0)
        else:
            color = cmap(norm(val))
        return f"background-color: {mcolors.rgb2hex(color)}"

    # Step 4: Apply the color scale to the DataFrame
    return df.style.map(color_scale)


def get_interest_space(array, path, sensibility):
    xmax, ymax = array.shape
    l_interest_area, potential_neighbors = [], []
    for x, y in path:
        potential_neighbors.extend(
            [
                (x + i, y + j)
                for i in range(-1 * sensibility, sensibility + 1)
                for j in range(-1 * sensibility, sensibility + 1)
            ]
        )
    for x_pot, y_pot in potential_neighbors:
        if x_pot >= 0 and x_pot < xmax and y_pot >= 0 and y_pot < ymax:
            l_interest_area.append((x_pot, y_pot))
    return l_interest_area


def visualize_room_path(room, sensibility, max_iter=0):
    path = room.is_playable_room(return_path=True, max_iter=max_iter)
    if type(path) == bool:
        return "No path has been found for this room."
    room_data_path = get_interest_space(room.data, path, sensibility)
    df_visu = pd.DataFrame(room.data)
    for tile_ in room_data_path:
        df_visu.loc[tile_] = "e"
    for tile in path:
        df_visu.loc[tile] = "p"
    return visualize_room(df_visu)


def extract_entity_coords(room, symbol):
    return [
        (x, y)
        for x, y in zip(
            np.where(room.data == symbol)[0], np.where(room.data == symbol)[1]
        )
    ]


def extract_non_lethal_entities_position(room):
    nle_pos = []
    l_nle = NL_ENTITES + ["1"]

    for ent in l_nle:
        nle_pos.extend(extract_entity_coords(room, ent))

    return nle_pos


def extract_non_lethal_entities_position_diff(room):
    nle_pos = []
    l_nle = NL_ENTITES + ["0", "1"]

    for ent in l_nle:
        nle_pos.extend(extract_entity_coords(room, ent))

    return nle_pos


def hole_presence(room, pos):
    """
    Return True if there is a hole below the current position
    """
    pot_hole = room.data[pos[0] :, pos[1]]
    return not ("1" in pot_hole or "_" in pot_hole or "D" in pot_hole)


def danger_presence(room, pos):
    pot_danger = room.data[pos[0] :, pos[1]]
    for sym in pot_danger:
        if sym == "0":
            continue
        elif sym == "^" or sym == "S":
            return True
        else:
            return False
    return True


def danger_density_normalized(room, path):
    """
    Return the density of danger within a path found by A*, normalized
    """
    l_danger = [danger_presence(room, pos) for pos in path]
    return sum(l_danger) / len(l_danger)


def evaluate_room_difficulty(room, path, sensibility):
    """
    Density of lethal entities + holes + emptiness
    Half-half contribution atm, can be weighted differently
    NB: density_diff + density_nle = 1
    """
    holes_diff = danger_density_normalized(room, path)
    # LE + void density: basically all that is not NLE
    zone_of_interest = get_interest_space(room.data, path, sensibility)
    entities_pos = extract_non_lethal_entities_position_diff(room)
    entities_pos_nle = extract_non_lethal_entities_position(room)

    nb_diff_zoi = len(zone_of_interest) - len(
        [pos for pos in entities_pos if pos in zone_of_interest]
    )
    nb_nle_zoi = len([pos for pos in entities_pos_nle if pos in zone_of_interest])

    local_density_diff = nb_diff_zoi / len(zone_of_interest)
    scarcity = len(zone_of_interest) / nb_nle_zoi

    return holes_diff, local_density_diff, scarcity


def get_entropy_data(data):
    d_sym = {}
    entropy = 0
    N = data.size
    for l in data:
        for j in l:
            if j not in d_sym.keys():
                d_sym[j] = 1
            else:
                d_sym[j] += 1

    for key in d_sym.keys():
        p = d_sym[key] / N
        entropy += -p * np.log10(p)

    return entropy


def evaluate_room_interestingness(room, path, sensibility):
    """
    Density of tiles + non-lethal entities
    Return the density of tiles + NLE in the whole room then
    the same quantity computed in the ZOI, noted as d_tot and
    s_int.
    0 < d_nle < 1: 0 - empty room besides LE; 1 - room full
    0 < s_int < 1: same but in ZOI
    """
    zone_of_interest = get_interest_space(room.data, path, sensibility)
    entities_pos = extract_non_lethal_entities_position(room)

    nb_nle_tot = len(entities_pos)
    nb_nle_zoi = len([pos for pos in entities_pos if pos in zone_of_interest])

    density_nle = nb_nle_tot / room.data.size
    interestingness_score = nb_nle_zoi / len(zone_of_interest)

    return density_nle, interestingness_score, get_entropy_data(room.data)


def get_coords_around_pos(room, pos, dist):
    def is_valid_coordinate(array, x, y):
        max_x = len(array)
        max_y = len(array[0]) if max_x > 0 else 0
        return 0 <= x < max_x and 0 <= y < max_y

    x, y = pos
    list_pos = []
    for side in (-1, 1):
        list_pos.extend([(x + side * dist, y + k) for k in range(-1 * dist, dist + 1)])
        list_pos.extend([(x + k, y + side * dist) for k in range(-1 * dist + 1, dist)])

    return [
        (a, b) for (a, b) in list_pos if is_valid_coordinate(room.data, a, b) and a >= x
    ]


def get_min_dist_to_nle(room, pos):
    l_nle = NL_ENTITES + ["1"]
    dist = 1
    if room.data[pos] in l_nle:
        return 0
    while dist < max(room.data.shape):
        l_pos_to_check = get_coords_around_pos(room, pos, dist)
        for pot_pos in l_pos_to_check:
            if room.data[pot_pos] in l_nle:
                return dist
        dist += 1
    # if not found: room has not a single NLE: not playable anyways
    raise ValueError("The room currently investigated has not a single NLE!")


def evaluate_astar_path(room, path):
    """
    Avg. distance to tiles + NL entities
    """
    l_dist_path_to_nle = []
    for path_pos in path:
        l_dist_path_to_nle.append(get_min_dist_to_nle(room, path_pos))

    return sum(l_dist_path_to_nle) / len(l_dist_path_to_nle)


def variance_astar_path(room, path):
    """
    Avg. distance to tiles + NL entities
    """
    l_dist_path_to_nle = []
    for path_pos in path:
        l_dist_path_to_nle.append(get_min_dist_to_nle(room, path_pos))

    return np.var(l_dist_path_to_nle)
