import bokeh
import bokeh.plotting
from bokeh.io import export_png
from bokeh.models import ColumnDataSource
import json
import numpy as np
import random

class Room():

    def __init__(self, width=1, height=1, origin_x=0, origin_y=0) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Width and height have to be positive!")
        self.size: list[int] = [width, height]
        self.origin: list[int] = [origin_x, origin_y]
        self.exits: dict[str, list[int]] = {
            "up":[],
            "down":[],
            "left":[],
            "right":[]
            }

    def get_size(self):
        return self.size
    
    def get_origin(self):
        return self.origin
    
    def get_exits(self):
        return self.exits
    
    def description(self):
        print(f"Bottom-left corner position: {self.origin}\nRoom size: {self.size}\nExits: {self.exits}")

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
        self.exits = {
            "up":[],
            "down":[],
            "left":[],
            "right":[]
            }
    
    def set_size(self, w, h):
        """
        Setter to modify size of a room. Might interfer with exits positions so you can only change size of a room when there are no exits.
        """
        exits = self.get_exits()
        for side in exits:
            if exits[side]: #at least one exit
                raise ValueError(f"At least one exit detected {side}, please use the clear_exits method if you want to change the size of the room.")
        if w <= 0 or h <= 0:
            raise ValueError("Width and height have to be positive!")
        self.size = [w, h]
    
    def set_origin(self, x, y):
        """
        Setter to modify the origin point (bottom left corner) of a room. Takes into account exits and move them accordingly.
        """
        old_x, old_y = self.get_origin()
        delta = [x-old_x, y-old_y]
        self.origin = [x, y]
        sides = ["up", "left", "down", "right"]
        for k in range(4):
            for exit in self.exits[sides[k]]:
                exit[0], exit[1] = exit[0] + delta[k%2], exit[1] + delta[k%2]
    
    def add_exit(self, side, c1, c2):
        if c1 >= c2:
            raise ValueError("Error: make sure coordinates are ordered and non equal")
        if side not in self.exits.keys():
            raise ValueError(f"Error: Select a side among {list(self.exits.keys())}")
        if side in ["up", "down"]:
            if c1 >= self.get_origin()[0] + 1 and c2 <= self.get_origin()[0] + self.get_size()[0] - 1:
                self.exits[side].append([c1, c2])
            else:
                raise ValueError(f"Exit out of bounds {(self.get_origin()[0] + 1, self.get_origin()[0] + self.get_size()[0] - 1)}!")
        else:
            if c1 >= self.get_origin()[1] + 1 and c2 <= self.get_origin()[1] + self.get_size()[1] - 1:
                self.exits[side].append([c1, c2])
            else:
                raise ValueError(f"Exit out of bounds {(self.get_origin()[1] + 1, self.get_origin()[1] + self.get_size()[1] - 1)}!")
    
    def generate_exit(self, side, size):
        if side in ["up", "down"]:
            c = np.random.randint(self.get_origin()[0] + 1, self.get_origin()[0] + self.get_size()[0] - size)
            self.add_exit(side, c, c+size)
        elif side in ["left", "right"]:
            c = np.random.randint(self.get_origin()[1] + 1, self.get_origin()[1] + self.get_size()[1] - size)
            self.add_exit(side, c, c+size)
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
                x0 = base_coord[k]*((k+1)%2) + (k%2)*exit[0]
                x1 = base_coord[k]*((k+1)%2) + (k%2)*exit[1]
                y0 = base_coord[k]*(k%2) + ((k+1)%2)*exit[0]
                y1 = base_coord[k]*(k%2) + ((k+1)%2)*exit[1]
                l_exits.append((x0, y0, x1, y1))
        return l_exits

    
    def is_overlapping(self, room):
        """
        Detects whether room is overlapping with another rectangle rect
        """
        R1 = self.get_origin() + self.get_tr_corner()
        R2 = room.get_origin() + room.get_tr_corner()
        if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
            return False
        else:
            return True

    def connect(self, room, side, skeleton):
        """
        Move self so that self is juxtaposed to room on the selected side
        """
        x0, y0 = room.get_origin()
    
        if side == "up":
            self.set_origin(np.random.randint(x0+3-self.get_size()[0], x0+room.get_size()[0]-2), y0+room.get_size()[1])
            ovl = skeleton.is_overlapping_with(self)
            if not ovl:
                x0_bis, y0_bis = self.get_origin()
                range_min, range_max = max(x0, x0_bis), min(x0+room.get_size()[0], x0_bis+self.get_size()[0])
                size = np.random.randint(1, range_max-range_min-1)
                c = np.random.randint(range_min+1, range_max-size)
                self.add_exit("down", c, c+size)
                room.add_exit("up", c, c+size)

        elif side == "down":
            self.set_origin(np.random.randint(x0+3-self.get_size()[0], x0+room.get_size()[0]-2), y0-self.get_size()[1])
            ovl = skeleton.is_overlapping_with(self)
            if not ovl:
                x0_bis, y0_bis = self.get_origin()
                range_min, range_max = max(x0, x0_bis), min(x0+room.get_size()[0], x0_bis+self.get_size()[0])
                size = np.random.randint(1, range_max-range_min-1)
                c = np.random.randint(range_min+1, range_max-size)
                self.add_exit("up", c, c+size)
                room.add_exit("down", c, c+size)

        elif side == "right":
            self.set_origin(x0+room.get_size()[0], np.random.randint(y0+3-self.get_size()[1], y0+room.get_size()[1]-2))
            ovl = skeleton.is_overlapping_with(self)
            if not ovl:
                x0_bis, y0_bis = self.get_origin()
                range_min, range_max = max(y0, y0_bis), min(y0+room.get_size()[1], y0_bis+self.get_size()[1])
                size = np.random.randint(1, range_max-range_min-1)
                c = np.random.randint(range_min+1, range_max-size)
                self.add_exit("left", c, c+size)
                room.add_exit("right", c, c+size)

        elif side == "left":
            self.set_origin(x0-self.get_size()[0], np.random.randint(y0+3-self.get_size()[1], y0+room.get_size()[1]-2))
            ovl = skeleton.is_overlapping_with(self)
            if not ovl:
                x0_bis, y0_bis = self.get_origin()
                range_min, range_max = max(y0, y0_bis), min(y0+room.get_size()[1], y0_bis+self.get_size()[1])
                size = np.random.randint(1, range_max-range_min-1)
                c = np.random.randint(range_min+1, range_max-size)
                self.add_exit("right", c, c+size)
                room.add_exit("left", c, c+size)
        else:
            raise ValueError(f"Selected side incorrect")


class Cskeleton():

    def __init__(self):
        self.lvl = dict()
        self.starting_room = None
        self.ending_room = None
    

    def description(self):
        if not self.lvl:
            print(f"The skeleton object is empty!")
        for room in self.lvl:
            print(f"Room name: {room}")
            self.lvl[room].description()
            print("\n")
    
    def get_nb_rooms(self):
        return len(list(self.lvl.keys()))
    
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
        for i in range(nb_rooms-1):
            for j in range(i+1, nb_rooms):
                if self.lvl[l_names_rooms[i]].is_overlapping(self.lvl[l_names_rooms[j]]):
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
        except:
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
        p = bokeh.plotting.figure()
        for room in self.lvl:
            x, y = self.lvl[room].get_origin()
            w, h = self.lvl[room].get_size()
            glyph = bokeh.models.Rect(x=x+w/2, y=y+h/2, width=w, height=h, fill_alpha=0.0, line_alpha=0.5)
            p.add_glyph(glyph)
            if show_names:
                source = ColumnDataSource(dict(x=[x+w/4], y=[y+h/2], text=[room]))
                glyph_txt = bokeh.models.Text(x="x", y="y", text="text", angle=0, text_color="#000000", text_font_size = {'value': '8px'})
                p.add_glyph(source, glyph_txt)
            l_exits = self.lvl[room].exits_to_list()
            for ex in l_exits:
                glyph = bokeh.models.Segment(x0=ex[0], y0=ex[1], x1=ex[2], y1=ex[3], line_color="#ff0000", line_width=2)
                p.add_glyph(glyph)
        if self.starting_room:
            x, y = self.lvl[self.starting_room].get_origin()
            w, h = self.lvl[self.starting_room].get_size()
            glyph = bokeh.models.Rect(x=x+w/2, y=y+h/2, width=w, height=h, fill_alpha=0.2, line_alpha=0.0, fill_color="#00ff1c")
            p.add_glyph(glyph)
        if self.ending_room:
            x, y = self.lvl[self.ending_room].get_origin()
            w, h = self.lvl[self.ending_room].get_size()
            glyph = bokeh.models.Rect(x=x+w/2, y=y+h/2, width=w, height=h, fill_alpha=0.2, line_alpha=0.0, fill_color="#ff00d4")
            p.add_glyph(glyph)
        bokeh.plotting.show(p)
        if save:
            fn = f"skeleton_{len(self.lvl)}.png"
            export_png(p, filename=fn)
    
    
    def to_JSON(self, filename):
        """
        Exports all data contained in the skeleton in a readable JSON
        """
        json_skeleton = {}

        for room in self.lvl:
            json_skeleton[room] = {
                "origin": self.lvl[room].origin,
                "size": self.lvl[room].size,
                "exits": self.lvl[room].exits,
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_skeleton, f, ensure_ascii=False, indent=4)
            f.close()

    

def PCG_skeleton(nb_rooms, p=0.5):
    """
    Generates procedurally a level skeleton made of nb_rooms rooms, 
    with a probability p of connecting the new room for each iteration, 
    instead of taking the last existing room.
    p = 0 generates a pathway style skeleton
    p > 0 generates a labyrinth style skeleton (if p close to 0: somewhat pathway-like)

    Returns a fully functionnal skeleton.
    """
    skeleton = Cskeleton()

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
            room = Room(random.randint(10, 100), random.randint(10, 100))
            room.connect(last_room, side, skeleton)
            overlapping = skeleton.is_overlapping_with(room)
        room_name = f"room_{k}"
        skeleton.add_room(room, room_name)
    
    skeleton.set_start_end()

    return skeleton
