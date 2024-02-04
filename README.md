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
