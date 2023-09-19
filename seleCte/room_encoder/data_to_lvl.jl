using Pkg
using Maple
using DelimitedFiles

# level_encoder function is the final step, meaning the room_set input
# needs to fulfill all requirements mentioned throughout the project

function level_encoder(level_title, room_set, output_filename)
    level = Map(
        level_title,
        room_set
    )

    encodeMap(level, output_filename)
end
