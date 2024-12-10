# seleCte Changelog

# [0.5.0] 10.12.2024

* Refactor the whole repo to make it public - remove tests, local exp., etc.

# [0.4.0] 05.08.2024

* Implementation of all evaluation functions and helpers in `utils.py`

# [0.3.0] 06.04.2024

* Implementation of A-star like algo and integration in the PCG generation for the sake of playability
* Fusion of both `celeste_pcg_utils.py` and `celeste_playability_utils.py` into a single `utils.py` file
* Pipeline to create a level based on CLI: from parameters to binary file

# [0.2.0] 02.02.2024

* Adapted the pcg model to handle the automated creation of borders and exits for each room
* Massive reorganization of the code
* Building pipeline part between celeskeleton and room encoder

# [0.1.0] 22.12.2023

* Extended notebook in pcg model to combine celeskeleton + room generation
* Map encoding function made in Julia - next step is the exits handling

# [0.0.8] 10.10.2023

* Basic draft of AI MdMC model
* Made visual interface v1 using pandas styler
* Recombination of fg + entities in a single structure to make it possible for the model to interpret it as a single object + creation of those maps

# [0.0.7] 22.09.2023

* Worked on room size distribution
    * Generated csv files with number of rooms per unique size, room sizes and entities data
    * Generated a few plots
    * Worked a lot in `celeste_data_exploration.ipynb` - needs a function for visual interface; Ahorn-like

# [0.0.6] 19.09.2023

* Updated organization of code for skeleton
    * Added a few functions to deal with interface skeleton/JSON
    * Minor fixes to `Room` and `Cskeleton` classes
* Slowly adapt from notebooks to Julia scripts

# [0.0.5] 07.03.2023

* Created data exploration functions
* Added first draft of playability exploration

# [0.0.4] 11.02.2023

* Finished skeleton v0

# [0.0.3] 07.02.2023

* Created the `Cskeleton` class for the skeleton part of the project

# [0.0.2] 04.02.2023

* Created the `Room` class for the skeleton part of the project

# [0.0.1] 03.02.2023

* Created the project structure
* Made bokeh visual interface for lvl skeleton
* Finished `data_loader` to build rooms database
