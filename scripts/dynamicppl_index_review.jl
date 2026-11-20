using AbstractMCMC: SamplingOutput
using AbstractPPL: AbstractPPL
using BangBang: setindex!!
using Distributions, DynamicPPL, ForwardDiff, Test, TOML
using DynamicPPL: predict
using DynamicPPL.VarNamedTuples: templated_setindex!!
using LinearAlgebra: I
using Random: Xoshiro
using MCMCChains: Chains

const HEAD = get(ENV, "REVIEW_HEAD", "false") == "true"
const OBSERVATIONS = Dict{String,Any}[]
function save_results()
    open("results.toml", "w") do io
        TOML.print(io, Dict("julia" => string(VERSION), "head" => HEAD, "observations" => OBSERVATIONS))
    end
end
atexit(save_results)
function observe(f, label)
    row = Dict{String,Any}("label" => label)
    try
        result = f()
        printable = result isa ParamsWithStats ? (params=collect(pairs(result.params)), stats=result.stats) : result
        row["value"] = repr(printable)
        row["type"] = string(typeof(result))
    catch err
        row["error"] = sprint(showerror, err)
        row["error_type"] = string(typeof(err))
    end
    push!(OBSERVATIONS, row)
    println(row)
    return row
end

@model function aliased_trips()
    trips = zeros(3, 3)
    trips[4] ~ Normal()
    return trips[4]
end
@model function sliced_trips()
    trips = zeros(3, 3)
    trips[:, :][4] ~ Normal()
    return trips[:, :][4]
end
@model function cartesian_trip()
    trips = zeros(3, 3)
    trips[1, 2] ~ Normal()
    return trips[1, 2]
end
ambiguous = VarNamedTuple((@varname(trips[1, 2]) => 1.0, @varname(trips[2, 2]) => 2.0))
@testset "PR body example and remedies" begin
    for (label, model) in (("direct", aliased_trips()), ("chained slice", sliced_trips()))
        for (opname, op) in (("returned", returned), ("logjoint", logjoint))
            observe("$label $opname") do
                op(model, ambiguous)
            end
        end
        observe("$label predict") do
            predict(Xoshiro(1), model, ambiguous; include_all=true)
        end
    end
    direct = observe("direct returned asserted") do
        returned(aliased_trips(), ambiguous)
    end
    @test HEAD ? get(direct, "error_type", "") == "ArgumentError" : direct["value"] == "2.0"
    template = zeros(3, 3)
    templated = templated_setindex!!(VarNamedTuple(), 1.0, @varname(trips[1, 2]), template)
    whole = copy(template)
    whole[1, 2] = 1.0
    for params in (templated, VarNamedTuple(; trips=whole), VarNamedTuple((@varname(trips[4]) => 1.0,)))
        @test returned(aliased_trips(), params) == 1.0
        @test logjoint(aliased_trips(), params) == logpdf(Normal(), 1.0)
    end
    observe("original templated VNT") do
        returned(aliased_trips(), templated)
    end
    observe("Dict of original VNT pairs") do
        returned(aliased_trips(), Dict(pairs(templated)))
    end
end

@testset "Index forms" begin
    for vn in (
        @varname(trips[4]), @varname(trips[1, 2]), @varname(trips[2, 2]),
        @varname(trips[CartesianIndex(1, 2)]),
        @varname(trips[[CartesianIndex(1, 2), CartesianIndex(2, 2)]]),
        @varname(trips[[false true; false true]]),
        @varname(trips[trues(2), 2]),
        @varname(trips[:, :][4]), @varname(trips[1:2, 1:2][4]),
        @varname(trips[1:3, :][4]), @varname(trips[:, 2][1]),
        @varname(trips[begin, end]), @varname(trips[1, 2, 1]),
    )
        for (opname, op) in (("haskey", haskey), ("getindex", getindex))
            observe("$opname $vn") do
                op(ambiguous, vn)
            end
        end
    end
    nested = VarNamedTuple((@varname(route.trips[1, 2]) => 1.0, @varname(route.trips[2, 2]) => 2.0))
    for vn in (@varname(route.trips[4]), @varname(route.trips[:, :][4]))
        observe("nested $vn") do
            nested[vn]
        end
    end
    for (name, value) in (("vector", [1.0, 2.0]), ("matrix", [1.0 2.0; 3.0 4.0]), ("view", view([1.0 2.0; 3.0 4.0], :, :)))
        vnt = VarNamedTuple(; trips=value)
        for vn in (@varname(trips[1]), @varname(trips[1, 1]), @varname(trips[1, 1, 1]))
            @test vnt[vn] == 1.0
        end
        observe("whole $name") do
            [vnt[@varname(trips[1])], vnt[@varname(trips[1, 1])]]
        end
    end
end

@model function trip_matrix(; linear=false)
    trips = zeros(3, 3)
    if linear
        for i in eachindex(trips)
            trips[i] ~ Normal(i, 2)
        end
    else
        for j in axes(trips, 2), i in axes(trips, 1)
            trips[i, j] ~ Normal(i + j, 2)
        end
    end
    total := sum(trips)
    return trips
end
@model function vector_trip()
    trips ~ MvNormal(zeros(3), I)
    return trips
end
@model function trip_parent(child)
    route ~ to_submodel(child)
    return route
end

@testset "Prediction workflows" begin
    for linear in (false, true)
        model = trip_matrix(; linear)
        template = zeros(3, 3)
        params = templated_setindex!!(VarNamedTuple(), 30.0, @varname(trips[1, 2]), template)
        params = templated_setindex!!(params, 40.0, @varname(trips[3, 3]), template)
        complete = predict(Xoshiro(23), model, params; include_all=true)
        selected = predict(Xoshiro(23), model, params)
        @test complete.stats == selected.stats
        @test complete.params[@varname(trips[1, 2])] == 30.0
        @test complete.params[@varname(trips[3, 3])] == 40.0
        @test !haskey(selected.params, @varname(trips[1, 2]))
        @test !haskey(selected.params, @varname(trips[3, 3]))
        for vn in (@varname(trips[1, 1]), @varname(trips[2, 3]), @varname(total))
            @test selected.params[vn] == complete.params[vn]
        end
        observe("matrix prediction linear=$linear") do
            (collect(keys(selected.params)), complete.params[@varname(trips)], selected.stats)
        end
        output = SamplingOutput(reshape([ParamsWithStats(params, (;)) for _ in 1:4], 2, 2); iterations=3:2:5)
        pred = predict(Xoshiro(8), model, output; multithreaded=false)
        again = predict(Xoshiro(8), model, output; multithreaded=false)
        @test pred.iterations == output.iterations
        @test all(isequal.(pred.samples, again.samples))
        threaded = predict(Xoshiro(8), model, output; multithreaded=true)
        threaded_again = predict(Xoshiro(8), model, output; multithreaded=true)
        @test all(isequal.(threaded.samples, threaded_again.samples))
        @test threaded.iterations == output.iterations
        chain = convert(Chains, pred)
        @test size(chain, 1) == 2
        observe("MCMCChains prediction linear=$linear") do
            result = predict(Xoshiro(8), model, chain)
            (size(result), range(result), names(result))
        end
    end
    linear = VarNamedTuple((@varname(trips[1]) => 30.0,))
    for model in (trip_matrix(; linear=true), condition(trip_matrix(; linear=true), @varname(trips[9]) => 50.0), fix(trip_matrix(; linear=true), @varname(trips[9]) => 50.0))
        result = predict(Xoshiro(2), model, linear)
        @test !haskey(result.params, @varname(trips[1]))
        @test haskey(result.params, @varname(trips[2]))
    end
    observe("nested submodel prediction") do
        model = trip_parent(trip_matrix(; linear=true))
        params = templated_setindex!!(VarNamedTuple(), 30.0, @varname(route.trips[1, 2]), VarNamedTuple(; trips=zeros(3, 3)))
        p = predict(Xoshiro(3), model, params)
        collect(keys(p.params))
    end
    observe("partial multivariate prediction") do
        predict(Xoshiro(4), vector_trip(), VarNamedTuple((@varname(trips[1]) => 1.0,)))
    end
    for model in (trip_matrix(), trip_matrix(; linear=true))
        expected = predict(Xoshiro(9), model, VarNamedTuple())
        unused = VarNamedTuple((@varname(unused[1, 2]) => 3.0,))
        @test isequal(expected, predict(Xoshiro(9), model, unused))
        missing_params = Matrix{Union{Missing,Float64}}(missing, 3, 3)
        missing_params[1, 2] = 30.0
        selected = predict(Xoshiro(9), model, VarNamedTuple(; trips=missing_params))
        @test !haskey(selected.params, @varname(trips[1, 2]))
        @test haskey(selected.params, @varname(trips[2, 2]))
    end
end

@testset "AD and type preservation" begin
    for T in (Float32, Float64, BigFloat)
        params = templated_setindex!!(VarNamedTuple(), T(1), @varname(trips[1, 2]), zeros(T, 3, 3))
        @test params[@varname(trips[1, 2])] isa T
        @test isequal(params[@varname(trips[1, 2])], T(1))
        @test returned(aliased_trips(), params) == T(1)
    end
    density(x) = logjoint(aliased_trips(), templated_setindex!!(VarNamedTuple(), x, @varname(trips[1, 2]), zeros(3, 3)))
    @test ForwardDiff.derivative(density, 1.0) == -1.0
end

observe("Julia API visibility") do
    [(name, Base.isexported(Base, name), isdefined(Base, :ispublic) ? Base.ispublic(Base, name) : false) for name in (:index_ndims, :to_indices, :ndims, :CartesianIndex)]
end
observe("AbstractPPL API exports") do
    [(name, Base.isexported(AbstractPPL, name)) for name in (:concretize_top_level, :varname_and_value_leaves, :getsym)]
end
save_results()
