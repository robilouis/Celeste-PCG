using CSV
using DataFrames
using DelimitedFiles
using JSON
using Maple

d_metadata_lvl = Dict()
open("../pcg/pcg_model_results/$lvl_name/data.json", "r") do f
    global d_metadata_lvl
    dicttxt = read(f, String)  # file information to string
    d_metadata_lvl = JSON.parse(dicttxt)  # parse and transform data
end
