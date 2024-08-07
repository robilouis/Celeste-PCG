import bokeh
import bokeh.plotting
from bokeh.io import export_png
from bokeh.models import ColumnDataSource
import itertools
import json
import logging
import os
import numpy as np
import pandas as pd
import random

import seleCte.utils as utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Room:

    def __init__(self, width=1, height=1, origin_x=0, origin_y=0, status=None) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Width and height have to be positive!")
        self.size: list[int] = [width, height]
        self.origin: list[int] = [origin_x, origin_y]
        self.exits: dict[str, list[int]] = {
            "up": [],
            "down": [],
            "left": [],
            "right": [],
        }
        self.data: np.array[object] = np.zeros(
            (self.size[1], self.size[0]), dtype=int
        ).astype(str)
        self.status = status

    def get_status(self):
        return self.status

    def get_size(self):
        return self.size

    def get_origin(self):
        return self.origin

    def get_exits(self):
        return self.exits

    def description(self):
        print(
            f"Bottom-left corner position: {self.origin}\nRoom size: {self.size}\nExits: {self.exits}\nData: {self.data}\nStatus: {self.status}"
        )

    def get_tr_corner(self):
        """
        Returns coordinates of top right corner of a room
        """
        return [self.origin[0] + self.size[0], self.origin[1] + self.size[1]]

    def get_corners(self):
        """
        Returns coordinates of bottom left and top right corner [x1, y1, x2, y2]
        """
        return self.get_origin() + self.get_tr_corner()

    def clear_exits(self):
        """
        Clears exits dict
        """
        self.exits = {"up": [], "down": [], "left": [], "right": []}

    def set_status(self, status):
        if status == "start" or status == "end":
            self.status = status
        else:
            raise ValueError("Room can only have start or end status, or None.")

    def set_size(self, w, h):
        """
        Setter to modify size of a room. Might interfer with exits positions so you can only change size of a room when there are no exits.
        """
        exits = self.get_exits()
        for side in exits:
            if exits[side]:  # at least one exit
                raise ValueError(
                    f"At least one exit detected {side}, please use the clear_exits method if you want to change the size of the room."
                )
        if w <= 0 or h <= 0:
            raise ValueError("Width and height have to be positive!")
        self.size = [w, h]

    def set_origin(self, x, y):
        """
        Setter to modify the origin point (bottom left corner) of a room. Takes into account exits and move them accordingly.
        """
        old_x, old_y = self.get_origin()
        delta = [x - old_x, y - old_y]
        self.origin = [x, y]
        sides = ["up", "left", "down", "right"]
        for k in range(4):
            for exit in self.exits[sides[k]]:
                exit[0], exit[1] = exit[0] + delta[k % 2], exit[1] + delta[k % 2]

    def set_exits(self, d_exits):
        self.exits = d_exits

    def set_data(self, data):
        if data.shape[0] == self.get_size()[1] and data.shape[1] == self.get_size()[0]:
            self.data = data
        else:  # mismatch in the size - should raise an error
            raise ValueError(
                f"Can't assign data of shape {data.shape} to a room of shape {(self.get_size()[1], self.get_size()[0])}!"
            )

    def add_exit(self, side, c1, c2):
        if c1 >= c2:
            raise ValueError("Error: make sure coordinates are ordered and non equal")
        if side not in self.exits.keys():
            raise ValueError(f"Error: Select a side among {list(self.exits.keys())}")
        if side in ["up", "down"]:
            if (
                c1 >= self.get_origin()[0] + 1
                and c2 <= self.get_origin()[0] + self.get_size()[0] - 1
            ):
                self.exits[side].append([c1, c2])
            else:
                raise ValueError(
                    f"Exit out of bounds {(self.get_origin()[0] + 1, self.get_origin()[0] + self.get_size()[0] - 1)}!"
                )
        else:
            if (
                c1 >= self.get_origin()[1] + 1
                and c2 <= self.get_origin()[1] + self.get_size()[1] - 1
            ):
                self.exits[side].append([c1, c2])
            else:
                raise ValueError(
                    f"Exit out of bounds {(self.get_origin()[1] + 1, self.get_origin()[1] + self.get_size()[1] - 1)}!"
                )

    def generate_exit(self, side, size):
        if side in ["up", "down"]:
            c = np.random.randint(
                self.get_origin()[0] + 1,
                self.get_origin()[0] + self.get_size()[0] - size,
            )
            self.add_exit(side, c, c + size)
        elif side in ["left", "right"]:
            c = np.random.randint(
                self.get_origin()[1] + 1,
                self.get_origin()[1] + self.get_size()[1] - size,
            )
            self.add_exit(side, c, c + size)
        else:
            raise ValueError(f"Error: Select a side among {list(self.exits.keys())}")

    def exits_to_list(self):
        """
        Returns the list of coordinates of the exits of the room as quadruplets (x0, y0, x1, y1)
        """
        l_exits = []
        sides = ["left", "down", "right", "up"]
        base_coord = self.get_corners()
        for k in range(4):
            for exit in self.exits[sides[k]]:
                x0 = base_coord[k] * ((k + 1) % 2) + (k % 2) * exit[0]
                x1 = base_coord[k] * ((k + 1) % 2) + (k % 2) * exit[1]
                y0 = base_coord[k] * (k % 2) + ((k + 1) % 2) * exit[0]
                y1 = base_coord[k] * (k % 2) + ((k + 1) % 2) * exit[1]
                l_exits.append((x0, y0, x1, y1))
        return l_exits

    def is_overlapping(self, room):
        """
        Detects whether room is overlapping with another rectangle rect
        """
        R1 = self.get_origin() + self.get_tr_corner()
        R2 = room.get_origin() + room.get_tr_corner()
        if (R1[0] >= R2[2]) or (R1[2] <= R2[0]) or (R1[3] <= R2[1]) or (R1[1] >= R2[3]):
            return False
        else:
            return True

    def connect(self, room, side, skeleton):
        """
        Move self so that self is juxtaposed to room on the selected side
        """
        x0, y0 = room.get_origin()

        if side == "up":
            self.set_origin(
                np.random.randint(
                    x0 + 3 - self.get_size()[0], x0 + room.get_size()[0] - 2
                ),
                y0 + room.get_size()[1],
            )
            ovl = skeleton.is_overlapping_with(self)
            if not ovl:
                x0_bis, y0_bis = self.get_origin()
                range_min, range_max = max(x0, x0_bis), min(
                    x0 + room.get_size()[0], x0_bis + self.get_size()[0]
                )
                size = np.random.randint(2, range_max - range_min - 1)
                c = np.random.randint(range_min + 1, range_max - size - 1)
                self.add_exit("down", c, c + size)
                room.add_exit("up", c, c + size)

        elif side == "down":
            self.set_origin(
                np.random.randint(
                    x0 + 3 - self.get_size()[0], x0 + room.get_size()[0] - 2
                ),
                y0 - self.get_size()[1],
            )
            ovl = skeleton.is_overlapping_with(self)
            if not ovl:
                x0_bis, y0_bis = self.get_origin()
                range_min, range_max = max(x0, x0_bis), min(
                    x0 + room.get_size()[0], x0_bis + self.get_size()[0]
                )
                size = np.random.randint(2, range_max - range_min - 1)
                c = np.random.randint(range_min + 1, range_max - size - 1)
                self.add_exit("up", c, c + size)
                room.add_exit("down", c, c + size)

        elif side == "right":
            self.set_origin(
                x0 + room.get_size()[0],
                np.random.randint(
                    y0 + 3 - self.get_size()[1], y0 + room.get_size()[1] - 2
                ),
            )
            ovl = skeleton.is_overlapping_with(self)
            if not ovl:
                x0_bis, y0_bis = self.get_origin()
                range_min, range_max = max(y0, y0_bis), min(
                    y0 + room.get_size()[1], y0_bis + self.get_size()[1]
                )
                size = np.random.randint(2, range_max - range_min - 1)
                c = np.random.randint(range_min + 1, range_max - size - 1)
                self.add_exit("left", c, c + size)
                room.add_exit("right", c, c + size)

        elif side == "left":
            self.set_origin(
                x0 - self.get_size()[0],
                np.random.randint(
                    y0 + 3 - self.get_size()[1], y0 + room.get_size()[1] - 2
                ),
            )
            ovl = skeleton.is_overlapping_with(self)
            if not ovl:
                x0_bis, y0_bis = self.get_origin()
                range_min, range_max = max(y0, y0_bis), min(
                    y0 + room.get_size()[1], y0_bis + self.get_size()[1]
                )
                size = np.random.randint(2, range_max - range_min - 1)
                c = np.random.randint(range_min + 1, range_max - size - 1)
                self.add_exit("right", c, c + size)
                room.add_exit("left", c, c + size)
        else:
            raise ValueError("Selected side incorrect")

    def get_starting_ending_points(self):
        l_points = []
        for exit in self.exits["up"]:
            l_points.append((0, int(np.mean(exit)) - self.origin[0]))
        for exit in self.exits["down"]:
            l_points.append((self.size[1] - 1, int(np.mean(exit)) - self.origin[0]))
        for exit in self.exits["left"]:
            l_points.append((self.origin[1] + int(np.mean(exit)), 0))
        for exit in self.exits["right"]:
            l_points.append((self.origin[1] + int(np.mean(exit)), self.size[0] - 1))

        return l_points

    def extract_astar_ready_data(self):
        rdy_data = pd.DataFrame(self.data)
        rdy_data = rdy_data.map(lambda x: "0" if x in ["D", "L", "P", "W"] else x)
        rdy_data = rdy_data.map(lambda x: "1" if x != "0" else x)
        rdy_data = rdy_data.to_numpy(dtype=int)
        return rdy_data

    def create_exits_in_matrix(self):
        def _create_exits(data, a, b, side, origin):
            height = data.shape[0]
            if side == "left":
                x, y = (a - origin[1]) - 1, (b - origin[1]) - 1
                data[x : y + 1, 0] = "0"
            elif side == "right":
                x, y = (a - origin[1]) - 1, (b - origin[1]) - 1
                data[x : y + 1, -1] = "0"
            elif side == "up":
                x, y = a - origin[0], b - origin[0]
                data[0, x : y + 1] = "0"
            elif side == "down":
                x, y = a - origin[0], b - origin[0]
                data[-1, x : y + 1] = "0"
            else:
                return KeyError
            return data

        l_exits = ["left", "right", "up", "down"]
        for exit in l_exits:
            for a, b in self.exits[exit]:
                self.data = _create_exits(
                    self.data, a, b, side=exit, origin=self.get_origin()
                )

    def is_playable_room(self, return_path=False, verbose=False, max_iter=0):
        # Preprocessing the room data
        room_astar_ready = self.extract_astar_ready_data()

        # Extracting exits ie. supposedly connected points
        room_exits_points = self.get_starting_ending_points()
        nb_exits = len(room_exits_points)

        if nb_exits > 1:  # not a dead end or a starting/ending room
            for k in range(nb_exits - 1):
                # use transitivity
                pt1, pt2 = room_exits_points[k], room_exits_points[k + 1]
                room_path = utils.astar(self, room_astar_ready, pt1, pt2, stop_condition=max_iter)
                if not room_path:
                    if verbose:
                        logger.warning("A* room playability failed in current room")
                    return False
        if verbose:
            logger.info("Playability check succeeded - current room seems playable!")
        if return_path:
            return room_path
        return True

    def is_valid_special(self, sp, return_path=False, verbose=False, max_iter=0):
        """
        Assert whether a special point (starting/ending) is valid,
        ie. is connected to all room exits
        """
        room_astar_ready = self.extract_astar_ready_data()
        room_exits_points = self.get_starting_ending_points()
        for pt in room_exits_points:
            sp_to_exits = utils.astar(self, room_astar_ready, sp, pt, stop_condition=max_iter)
            if not sp_to_exits:
                if verbose:
                    logger.warning("Special point cannot reach all room exits")
                return False
        if verbose:
            logger.info("Playability check succeeded - special point is valid")
        if return_path:
            return sp_to_exits
        return True

    def add_respawn_points(self, clear_space_size=2):
        """
        Add player entities near entries to enable room access
        """
        offsets = {
            "up": (np.array([2, 0]), 1, 0),
            "down": (np.array([self.size[1] - 3, 0]), 1, 0),
            "left": (np.array([0, 2]), 0, 1),
            "right": (np.array([0, self.size[0] - 3]), 0, 1),
        }
        for side in ["up", "down", "left", "right"]:
            for exit in self.exits[side]:
                respawn_xy = offsets[side][0]
                respawn_xy[offsets[side][1]] = int(
                    np.mean(exit) - self.origin[offsets[side][2]]
                )
                self.data[respawn_xy[0], respawn_xy[1]] = "P"
                for i, j in [
                    (a, b)
                    for a in range(-1 * clear_space_size + 1, clear_space_size)
                    for b in range(-1 * clear_space_size + 1, clear_space_size)
                    if (a, b) != (0, 0)
                ]:
                    self.data[respawn_xy[0] + i, respawn_xy[1] + j] = "0"

    def add_special_points(self, nb_tries_limit, verbose=False):
        """
        Add structure with player spawn point if starting room or crystal heart if
        ending room - returns a boolean whether it worked within nb_tries_limit tries
        """
        if not self.status:
            if verbose:
                logger.info(
                    "This room is neither a starting not an ending room - skipping."
                )
            return True

        valid_sp = False
        try_iter = 0
        while (
            not valid_sp and try_iter < nb_tries_limit
        ):  # avoids softlock in the middle of lvl gen
            try_iter += 1
            random_sp = tuple(
                np.random.randint([self.size[1] - 4, self.size[0] - 3])
                + np.array([3, 1])
            )
            valid_sp = self.is_valid_special(random_sp)

        if not valid_sp:  # something is probably wrong with room - need re-gen
            if verbose:
                logger.warning(
                    f"Could not find a special point in {nb_tries_limit} iterations - re-generating room."
                )
            return False

        if verbose:
            logger.info(f"Special point found in {try_iter} iterations.")
        if self.status == "end":
            self.data[random_sp[0], random_sp[1]] = "H"
        elif self.status == "start":
            self.data[random_sp[0], random_sp[1] - 1 : random_sp[1] + 2] = "D"
            self.data[random_sp[0] - 1, random_sp[1]] = "P"
        else:
            raise ValueError("Status should be either None, 'start' or 'end'!")
        return True
    
    def save(self, path_to_save):
        """
        Exports all data contained in the skeleton in a readable JSON
        """

        json_room = {
            "origin": self.origin,
            "size": self.size,
            "exits": self.exits,
            "data": self.data
        }
        np.savetxt(
            f"{path_to_save}.csv",
            self.data,
            fmt="%s",
        )

        with open(
            f"{path_to_save}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(json_room, f, ensure_ascii=False, indent=4)
            f.close()

        logger.info(
            f"Saved lvl data in folder {path_to_save}"
        )


class Cskeleton:

    def __init__(self):
        self.lvl = dict()
        self.starting_room = None
        self.ending_room = None

    def list_all_rooms(self):
        return list(self.lvl.keys())

    def description(self):
        if not self.lvl:
            print("The skeleton object is empty!")
        for room in self.lvl:
            print(f"Room name: {room}")
            self.lvl[room].description()
            print("\n")

    def get_nb_rooms(self):
        return len(list(self.lvl.keys()))

    def get_starting_rooms(self):
        return [room for room in self.lvl if self.lvl[room].get_status() == "start"]

    def get_ending_rooms(self):
        return [room for room in self.lvl if self.lvl[room].get_status() == "end"]

    def add_room(self, room, name):
        if name in self.lvl.keys():
            raise ValueError(f"Room {name} already exists! Please use another name.")
        self.lvl[name] = room

    def remove_room(self, name):
        if name in self.lvl.keys():
            del self.lvl[name]
        else:
            raise ValueError(f"Room {name} does not exists!")

    def global_overlapping(self):
        nb_rooms = len(self.lvl)
        l_names_rooms = list(self.lvl.keys())
        l_index_overlaps = []
        for i in range(nb_rooms - 1):
            for j in range(i + 1, nb_rooms):
                if self.lvl[l_names_rooms[i]].is_overlapping(
                    self.lvl[l_names_rooms[j]]
                ):
                    l_index_overlaps.append((l_names_rooms[i], l_names_rooms[j]))
        return l_index_overlaps

    def is_overlapping_with(self, room):
        for name in self.lvl:
            if room.is_overlapping(self.lvl[name]):
                return True
        return False

    def get_room_by_name(self, name):
        """
        Returns room by name, will modify the original obj even if different name
        """
        try:
            return self.lvl[name]
        except KeyError:
            raise KeyError(f"Invalid key: room {name} does not exist!")

    def select_last_room(self, room_name, p):
        """
        Returns last room, with a proba p of selecting a random room in the skeleton
        """
        if np.random.rand() < p:
            name = np.random.choice(list(self.lvl.keys()))
        else:
            name = room_name
        return self.get_room_by_name(name)

    def set_start_end(self):
        """
        Sets starting_room and ending_room attributes of a skeleton, comes handy later when generating playable level.
        """
        l_lvl_names = list(self.lvl.keys())
        self.starting_room, self.ending_room = l_lvl_names[0], l_lvl_names[-1]

    def show_skeleton(self, show_names=True, save=True):
        """
        Shows a super cool plot of the skeleton, underlining start and end if existing
        """
        p = bokeh.plotting.figure(match_aspect=True)
        for room in self.lvl:
            x, y = self.lvl[room].get_origin()
            w, h = self.lvl[room].get_size()
            glyph = bokeh.models.Rect(
                x=x + w / 2,
                y=y + h / 2,
                width=w,
                height=h,
                fill_alpha=0.0,
                line_alpha=0.5,
            )
            p.add_glyph(glyph)
            if show_names:
                source = ColumnDataSource(
                    dict(x=[x + w / 4], y=[y + h / 2], text=[room])
                )
                glyph_txt = bokeh.models.Text(
                    x="x",
                    y="y",
                    text="text",
                    angle=0,
                    text_color="#000000",
                    text_font_size={"value": "8px"},
                )
                p.add_glyph(source, glyph_txt)
            l_exits = self.lvl[room].exits_to_list()
            for ex in l_exits:
                glyph = bokeh.models.Segment(
                    x0=ex[0],
                    y0=ex[1],
                    x1=ex[2],
                    y1=ex[3],
                    line_color="#ff0000",
                    line_width=2,
                )
                p.add_glyph(glyph)
        if self.starting_room:
            x, y = self.lvl[self.starting_room].get_origin()
            w, h = self.lvl[self.starting_room].get_size()
            glyph = bokeh.models.Rect(
                x=x + w / 2,
                y=y + h / 2,
                width=w,
                height=h,
                fill_alpha=0.2,
                line_alpha=0.0,
                fill_color="#00ff1c",
            )
            p.add_glyph(glyph)
        if self.ending_room:
            x, y = self.lvl[self.ending_room].get_origin()
            w, h = self.lvl[self.ending_room].get_size()
            glyph = bokeh.models.Rect(
                x=x + w / 2,
                y=y + h / 2,
                width=w,
                height=h,
                fill_alpha=0.2,
                line_alpha=0.0,
                fill_color="#ff00d4",
            )
            p.add_glyph(glyph)
        bokeh.plotting.show(p)
        if save:
            fn = f"skeleton_{len(self.lvl)}.png"
            export_png(p, filename=fn)

    def save(self, lvl_name):
        """
        Exports all data contained in the skeleton in a readable JSON
        """
        json_skeleton = {}
        os.makedirs(f"./seleCte/pcg/pcg_model_results/{lvl_name}/", exist_ok=True)

        for room in self.lvl:
            json_skeleton[room] = {
                "origin": self.lvl[room].origin,
                "size": self.lvl[room].size,
                "exits": self.lvl[room].exits,
            }
            np.savetxt(
                f"./seleCte/pcg/pcg_model_results/{lvl_name}/{room}.csv",
                self.lvl[room].data,
                fmt="%s",
            )

        with open(
            f"./seleCte/pcg/pcg_model_results/{lvl_name}/data.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(json_skeleton, f, ensure_ascii=False, indent=4)
            f.close()

        logger.info(
            f"Saved lvl data in folder ./seleCte/pcg/pcg_model_results/{lvl_name}"
        )

    def format_filled_celeskeleton(self):
        def _create_full_borders_room(data, tile_id=1):
            data[0, :] = tile_id
            data[-1, :] = tile_id
            data[:, 0] = tile_id
            data[:, -1] = tile_id

            return data

        for room_name in self.list_all_rooms():
            room = self.get_room_by_name(room_name)
            room.set_data(_create_full_borders_room(room.data))
            room = room.create_exits_in_matrix()

    # # TODO: adapt to 1-exit rooms for space assertion once the strawberry spawner is done
    # def is_playable(self, return_paths=False):
    #     l_paths = []
    #     for i in range(len(self.list_all_rooms())):
    #         temp_room = self.get_room_by_name(f"room_{i+1}")
    #         temp_exits_points = temp_room.get_starting_ending_points()
    #         temp_map = temp_room.extract_astar_ready_data()
    #         if len(temp_exits_points) > 1:  # not a dead end or a starting/ending room
    #             for pt1, pt2 in itertools.combinations(
    #                 temp_exits_points, 2
    #             ):  # make sure all exits are reachable
    #                 temp_path = utils.astar(temp_map, pt1, pt2)
    #                 l_paths.append((f"room_{i+1}", temp_path))
    #                 if not temp_path:
    #                     logger.warning(f"A* room playability failed in room {i+1}")
    #                     return False

    #     logger.info("Playability check succeeded!")
    #     if return_paths:
    #         return l_paths
    #     return True

    def extract_rooms_metadata(self):
        rooms_metadata = []
        for room_name in self.list_all_rooms():
            room = self.get_room_by_name(room_name)
            rooms_metadata.append((room.size, room.origin, room.exits))

        return rooms_metadata


def PCG_skeleton(nb_rooms, p=0.5, size=None):
    """
    Generates procedurally a level skeleton made of nb_rooms rooms,
    with a probability p of connecting the new room for each iteration,
    instead of taking the last existing room.
    p = 0 generates a pathway style skeleton
    p > 0 generates a labyrinth style skeleton (if p close to 0: somewhat pathway-like)

    Returns a fully functionnal skeleton.
    """
    skeleton = Cskeleton()

    if size:
        size_init = size
    else:
        size_init = [random.randint(10, 100) for _ in range(2)]
    room_init = Room(size_init[0], size_init[1])
    sides = ["up", "down", "left", "right"]

    k = 1
    room_name = f"room_{k}"

    skeleton.add_room(room_init, room_name)

    while skeleton.get_nb_rooms() < nb_rooms:
        k += 1
        overlapping = True
        while overlapping:
            last_room = skeleton.select_last_room(room_name, p)
            side = np.random.choice(sides)
            if size:
                room = Room(size[0], size[1])
            else:
                room = Room(random.randint(10, 100), random.randint(10, 100))
            try:
                room.connect(last_room, side, skeleton)
            except ValueError:
                continue
            overlapping = skeleton.is_overlapping_with(room)
        room_name = f"room_{k}"
        skeleton.add_room(room, room_name)

    skeleton.set_start_end()
    skeleton.lvl[skeleton.starting_room].set_status("start")
    skeleton.lvl[skeleton.ending_room].set_status("end")

    return skeleton


def json_to_skeleton(data_json):
    """
    Extracts data from json file and returns Cskeleton object
    """
    json_skeleton = Cskeleton()
    for name in data_json:
        w, h = data_json[name]["size"]
        x, y = data_json[name]["origin"]
        room = Room(width=w, height=h, origin_x=x, origin_y=y)
        room.set_exits(data_json[name]["exits"])
        json_skeleton.add_room(room, name)

    json_skeleton.set_start_end()

    return json_skeleton


def load_data_to_celeskeleton(folder_path):
    with open(os.path.join(folder_path, "data.json")) as f:
        json_file = json.load(f)
        f.close()
    skel = json_to_skeleton(json_file)
    for i in range(len(skel.list_all_rooms())):
        temp_data = pd.read_csv(
            os.path.join(folder_path, f"room_{i+1}.csv"),
            header=None,
            sep=" ",
            dtype=str,
        ).to_numpy()
        skel.get_room_by_name(f"room_{i+1}").set_data(temp_data)

    return skel
