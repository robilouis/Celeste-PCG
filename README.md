# Celeste-PCG
Master Thesis on PCG applied to level generation for the platformer Celeste

## Data loader
Loads data from the game, and makes a usable database

## Cskeleston
Generate structure of levels

## Playability
Model assessing the clearability of a room

## Room Generator
Model generating content procedurally to create rooms

## Julia Encoder function
Use the cskeleston, the room generator and the playability module to generate final levels
#### Requirements
- Level skeleton $L = \{l_1, \ldots, l_n\}$: set of empty rooms indexed by an integer $i$, their sizes $(h_i, w_i)$ and their spatial coordinates $(x_i, y_i)$ 
- A set of generated rooms $R = \{r_1, \ldots, r_K\}$
- Playability module
#### Procedure
For each empty room $l_i$, we match a room $r_\phi(i)$ such that:
- $r_\phi(i)$ has the right dimensions
- $r_\phi(i)$ is playable
- Exits somewhat match -> maybe post-processing, same for starting/ending room.