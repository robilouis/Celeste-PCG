# Celeste-PCG

Hey hey - welcome to this Master Thesis project on PCG applied to level generation for the platformer Celeste. Here is a short description of the modules composing this repository. Doc is still a WIP though, good luck if you wanna dive into this!

## Cskeleston

Generate structure of levels. Contains both essential classes for PCG generation applied to Celeste:
- Cskeleton: global, high-level "skeleton" of a Celeste level - contains tunable generation function, display function (using bokeh) and saving/loading options
- Room: object containing the data of a room, sub-part of a level. Contains various setter, getter, check, generation, visualization, and evaluation functions.

## Data

#### Levels
Celeste game data split per level - 3 files per ROOM:
- foreground (x_fg.csv)
- background (x_bg.csv)
- entities (x_entities.json)

No binary game files in there - only a db of exploitable tiles & entities. Please add your own .bin files there if you want to work with more/different data (mods like Strawberry Jam for example)! 

#### Data Exploration
A few files gathering basic data about rooms in Celeste.

## Data loader

Contains a `Julia` notebook which loads data from the game binary files, and makes a usable database from them. Useful if you want to work with extra maps that are not in the original game (mods). Please note that this notebook makes use of `Maple` (https://github.com/CelestialCartographers/Maple), you might want to add/adapt entities (e.g. in `entities.jl` and/or `tiles.jl`) to read/generate custom maps.

## Level Generation

Use the celeskeleton, the PCG and the playability module to generate final levels from start to end.

## PCG

All the scripts and files related to room generation, based on a Multi-dimensional Markov Chains model trained on the game data.
The scripts `x_generation.py` can be used to generate room batches, full levels, and PE (probabilities estimation) dictionary based on a single command line.

## Playability

Model assessing the clearability of a room, based on a tweaked version of the A* algorithm. Please note that this is a WIP and does not reflect the true playability of a room, only gives an indication. Most of the code related to it can be found in `utils.py`.

## Room Encoder

Contains a notebook to get familiar with how `Maple` works when it comes to encoding data into playable `Celeste` binary files, and a `Julia` script encoding output of the pcg model into a playable .bin file.


## TODOs/potential improvements:
- Add small functions for designing beginning/respawn/ending points of a generated level (check if heart is already implemented in Maple) => could be improved by going through room until a given distance of nb_iter is reached: avoid randomness and helps with gen time
- Adapt PCG model to generate interesting strawberries/collectibles
- Add parameters in the Celeskeleton module to emulate global generation direction of a lvl: if None, should be like a random generation as it is atm. Should provide a direction (out of 8) and a "strength" parameter, from 0 to 1; 0 meaning the direction is basically ignored, 1 is like strictly respected (should be strictly between 0 and 1).
- Revision of the base PCG model, we do not want to generate fully closed rooms, or at least give the possibility not to do so - change in model.py (remove walls and ceiling with post-processing e.g.)
- Focus on playability, interestingness and difficulty