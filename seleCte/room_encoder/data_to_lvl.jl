using CSV
using DataFrames
using DelimitedFiles
using JSON
using Pkg

Pkg.add(PackageSpec(url="https://github.com/CelestialCartographers/Maple.git"))  # change this if you forked Maple to make some modifications

using Maple


function get_origin_room_from_metadata_dict(room_nb, d_metadata)
    get(get(d_metadata, "room_$room_nb", nothing), "origin", nothing)
end


function create_room(room_nb, d_metadata, lvl_data_folder)
    data = Matrix(CSV.read("$lvl_data_folder/room_$room_nb.csv", DataFrame, header=false, types=Char))
    fgTiles = FgTiles(data)
    origin = get_origin_room_from_metadata_dict(room_nb, d_metadata)
    Room(
        name="room_$room_nb",
        fgTiles=fgTiles,
        position=(origin[1] * 8, origin[2] * -8),
        size=size(fgTiles),
        entities=entityMap(data)
    )
end


function create_map_file(d_metadata, lvl_data_folder, map_name)
    room_vect = Room[]
    for i = 1:length(d_metadata)
        push!(room_vect, create_room(i, d_metadata, lvl_data_folder))
    end
    map = Map(
        map_name,
        room_vect
    )

    encodeMap(map, "./seleCte/room_encoder/generated_bin_files/$map_name.bin")
end


lvl_folder_name = ARGS[1]
if size(ARGS)[1] == 2
    output_bin_file_name = ARGS[2]
else
    output_bin_file_name = "default_map"
end


d_metadata_lvl = Dict()
open("./seleCte/pcg/pcg_model_results/$lvl_folder_name/data.json", "r") do f
    global d_metadata_lvl
    dicttxt = read(f, String)  # file information to string
    d_metadata_lvl = JSON.parse(dicttxt)  # parse and transform data
end

create_map_file(d_metadata_lvl, "./seleCte/pcg/pcg_model_results/$lvl_folder_name", output_bin_file_name)
println("Done!")
