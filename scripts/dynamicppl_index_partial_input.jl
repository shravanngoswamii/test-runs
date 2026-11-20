using DynamicPPL, Distributions, Test, TOML
using DynamicPPL: predict
using LinearAlgebra: I
using Random: Xoshiro

const HEAD = get(ENV, "REVIEW_HEAD", "false") == "true"
const RESULTS = Dict{String,Any}[]
@model function first_route()
    speeds = Vector{Vector{Float64}}(undef, 2)
    speeds[1] ~ MvNormal(zeros(1), I)
    arrival ~ Normal(speeds[1][1], 1)
    return arrival
end
@model function arrival_only()
    arrival ~ Normal()
    return arrival
end

@testset "Prediction only reads supplied model entries" begin
    partial = Vector{Vector{Float64}}(undef, 2)
    partial[1] = [30.0]
    for (label, model, params) in (
        ("partly assigned input", first_route(), VarNamedTuple(; speeds=partial)),
        ("unused array input", arrival_only(), VarNamedTuple(; unused=partial)),
        ("fully assigned control", first_route(), VarNamedTuple(; speeds=[[30.0], [40.0]])),
    )
        complete = predict(Xoshiro(1), model, params; include_all=true)
        @test haskey(complete.params, @varname(arrival))
        row = Dict{String,Any}("label" => label)
        try
            selected = predict(Xoshiro(1), model, params)
            row["keys"] = string.(collect(keys(selected.params)))
            row["arrival"] = selected.params[@varname(arrival)]
            @test selected.params[@varname(arrival)] == complete.params[@varname(arrival)]
            @test selected.stats == complete.stats
        catch err
            row["error_type"] = string(typeof(err))
            row["error"] = sprint(showerror, err)
        end
        push!(RESULTS, row)
        println(row)
        if label == "fully assigned control" || !HEAD
            @test !haskey(row, "error")
        else
            @test get(row, "error_type", "") == "UndefRefError"
        end
        @test !isassigned(partial, 2)
    end
end
open("results.toml", "w") do io
    TOML.print(io, Dict("head" => HEAD, "julia" => string(VERSION), "observations" => RESULTS))
end
