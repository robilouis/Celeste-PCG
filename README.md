# Celeste-PCG

Master Thesis on PCG applied to level generation for the platformer Celeste

## Cskeleston

Generate structure of levels

## Data

Celeste game data

## Data loader

Loads data from the game, and makes a usable database

## Level Generation

Use the cskeleston, the PCG and the playability module to generate final levels from start to end

## PCG

All the scripts and files related to room generation, based on a Multi-dimensional Markov Chains model trained on the game data

## Playability

Model assessing the clearability of a room, based on a tweaked version of the A* algorithm

## Room Encoder

Model generating content procedurally to create rooms

#### Misc
Script for formatting `bash -c 'pyupgrade --py311-plus $(find **/*.py) && black . && ruff . --fix '`

# TODOs:
- Add small functions for designing beginning/respawn/ending points of a generated level (check if heart is already implemented in Maple)
- Add parameters to orient generation direction of a lvl: if None, should be like a random generation as it is atm. Should provide a direction (out of 8) and a "strength" parameter, from 0 to 1; 0 meaning the direction is basically ignored, 1 is like strictly respected (should be strictly between 0 and 1).
- Revision of the base pcg model, we do not want to generate fully closed rooms, or at least give the possibility not to do so
- Focus on playability, interestingness and complexity