using AbstractMCMC: SamplingOutput
using ADTypes: AutoForwardDiff
using BangBang: empty!!
using Distributions
using DynamicPPL
using ForwardDiff
using LogDensityProblems: logdensity, logdensity_and_gradient
using Random: Xoshiro
using Serialization
using Test
using TOML

const DP = DynamicPPL
const REQUIRE_RECORDED = get(ENV, "EXPECT_RECORDED_QUERIES", "false") == "true"
const RESULTS = Dict{String,Any}("julia" => string(VERSION), "changed_queries" => REQUIRE_RECORDED)
const QUERIES = (keys, values, length, isempty, vi -> haskey(vi, @varname(usual_minutes)), is_transformed)

function check_unrecorded(vi)
    for (query, old) in zip(QUERIES, ((), (), 0, true, false, false))
        if REQUIRE_RECORDED
            @test_throws ArgumentError query(vi)
        else
            @test query(vi) == old
        end
    end
    @test !is_transformed(vi, VarName[])
    @test_throws ArgumentError is_transformed(vi, @varname(usual_minutes))
    @test_throws ArgumentError is_transformed(vi, [@varname(usual_minutes)])
end

function check_recorded(vi, expected)
    @test Set(keys(vi)) == Set(expected)
    @test length(vi) == length(expected)
    @test isempty(vi) == isempty(expected)
    @test length(values(vi)) == length(expected)
    @test !haskey(vi, @varname(absent))
    @test !is_transformed(vi, VarName[])
    for vn in expected
        @test haskey(vi, vn)
        @test_throws KeyError DP.get_transformed_value(vi, @varname(absent))
    end
end

@model function commute(observations)
    usual_minutes ~ Normal(35, 8)
    day_variation ~ LogNormal(1, 0.25)
    for day in eachindex(observations)
        observations[day] ~ Normal(usual_minutes, day_variation)
    end
    return usual_minutes, day_variation
end

@model function threaded_readings(readings)
    usual_minutes ~ Normal(35, 8)
    Threads.@threads for day in eachindex(readings)
        readings[day] ~ Normal(usual_minutes, 3)
    end
    return usual_minutes
end

@model no_variables() = nothing
@model observation_only() = 1.0 ~ Normal()
@model function branching(active)
    usual_minutes ~ Normal()
    if active
        extra_time ~ Exponential()
    end
    return usual_minutes
end

@testset "VarInfo query review" begin
    @testset "absent recorder" begin
        for make in (() -> VarInfo(), () -> VarInfo(()), () -> VarInfo(RawValueAccumulator(false)))
            for wrap in (identity, DP.ThreadSafeVarInfo)
                check_unrecorded(wrap(make()))
            end
        end
    end

    @testset "empty recorder" begin
        for model in (no_variables(), observation_only())
            for wrap in (identity, DP.ThreadSafeVarInfo)
                vi = wrap(VarInfo(VectorValueAccumulator(), DP.default_accumulators()...))
                _, vi = init!!(Xoshiro(1), model, vi, InitFromPrior(), UnlinkAll())
                check_recorded(vi, VarName[])
                @test !is_transformed(vi)
            end
        end
        vi = VarInfo(VectorValueAccumulator())
        @test isempty(@inferred keys(vi))
        @test isempty(@inferred values(vi))
        @test (@inferred length(vi)) == 0
        @test @inferred isempty(vi)
        @test !(@inferred haskey(vi, @varname(absent)))
        @test !(@inferred is_transformed(vi))
    end

    @testset "evaluation and migration" begin
        observations = [28.0, 31.0, 29.0]
        model = commute(observations)
        params = (; usual_minutes=30.0, day_variation=2.0)
        strategy = InitFromParams(params, nothing)
        expected_prior = logpdf(Normal(35, 8), 30.0) + logpdf(LogNormal(1, 0.25), 2.0)
        expected_likelihood = sum(x -> logpdf(Normal(30, 2), x), observations)
        for transform in (UnlinkAll(), LinkAll())
            for wrap in (identity, DP.ThreadSafeVarInfo)
                _, unrecorded = init!!(Xoshiro(2), model, wrap(VarInfo()), strategy, transform)
                check_unrecorded(unrecorded)
                @test getlogprior(unrecorded) ≈ expected_prior
                @test getloglikelihood(unrecorded) ≈ expected_likelihood
                _, recorded = init!!(
                    Xoshiro(2), model,
                    wrap(VarInfo(VectorValueAccumulator(), DP.default_accumulators()...)),
                    strategy, transform,
                )
                check_recorded(recorded, [@varname(usual_minutes), @varname(day_variation)])
                @test is_transformed(recorded) == (transform isa LinkAll)
                @test is_transformed(recorded, [@varname(day_variation)]) == (transform isa LinkAll)
                @test getlogp(recorded) == getlogp(unrecorded)
                @test getlogjoint(recorded) ≈ expected_prior + expected_likelihood
                snapshot = deepcopy(get_vector_values(recorded))
                for _ in 1:3
                    check_recorded(recorded, keys(snapshot))
                end
                @test isequal(get_vector_values(recorded), snapshot)
            end
        end
        vi = VarInfo(Xoshiro(3), model, strategy)
        @test length(@inferred keys(vi)) == 2
        @test length(@inferred values(vi)) == 2
        @test (@inferred length(vi)) == 2
        @test !(@inferred isempty(vi))
        @test @inferred haskey(vi, @varname(usual_minutes))
        @test !(@inferred is_transformed(vi))
        @test (@inferred VarInfo(Xoshiro(3), model, strategy)) == vi
        @test isempty(empty!!(copy(vi)))
        @test length(vi) == 2
        @test keys(subset(vi, [@varname(usual_minutes)])) == [@varname(usual_minutes)]
        buffer = IOBuffer()
        serialize(buffer, vi)
        seekstart(buffer)
        @test isequal(deserialize(buffer), vi)
        _, raw = init!!(Xoshiro(3), model, VarInfo(RawValueAccumulator(false)), strategy)
        check_unrecorded(raw)
        @test get_raw_values(raw)[@varname(usual_minutes)] == 30.0
        @test ParamsWithStats(raw).params[@varname(day_variation)] == 2.0
        RESULTS["logjoint"] = getlogjoint(vi)
        RESULTS["values"] = internal_values_as_vector(vi)

        ldf = LogDensityFunction(model, getlogjoint_internal, vi; adtype=AutoForwardDiff())
        density, gradient = logdensity_and_gradient(ldf, [30.0, 2.0])
        @test density ≈ expected_prior + expected_likelihood
        @test gradient ≈ ForwardDiff.gradient(
            x -> logjoint(model, (; usual_minutes=x[1], day_variation=x[2])), [30.0, 2.0]
        )
        @test logdensity(ldf, [30.0, 2.0]) ≈ density
        RESULTS["gradient"] = gradient

        draws = SamplingOutput(reshape([ParamsWithStats(raw), ParamsWithStats(raw)], 2, 1); iterations=3:2:5)
        heldout = commute(fill(missing, 2))
        predictions = predict(Xoshiro(4), heldout, draws)
        repeated = predict(Xoshiro(4), heldout, draws)
        @test predictions.iterations == draws.iterations
        @test predictions.samples == repeated.samples
        @test all(ps -> haskey(ps.params, @varname(observations[1])), predictions.samples)
        RESULTS["predictions"] = [ps.params[@varname(observations[1])] for ps in vec(predictions.samples)]
        pointwise = pointwise_loglikelihoods(model, strategy)
        @test sum(sum, values(pointwise)) ≈ expected_likelihood
        @test observations == [28.0, 31.0, 29.0]
    end

    @testset "threaded evaluation without and with recording" begin
        model = setthreadsafe(threaded_readings([28.0, 31.0, 29.0]), true)
        strategy = InitFromParams((; usual_minutes=30.0), nothing)
        _, raw = init!!(Xoshiro(5), model, VarInfo(), strategy)
        check_unrecorded(raw)
        vi = VarInfo(Xoshiro(5), model, strategy)
        check_recorded(vi, [@varname(usual_minutes)])
        @test getlogjoint(vi) ≈ getlogjoint(raw)
        @test ForwardDiff.derivative(x -> logjoint(model, (; usual_minutes=x)), 30.0) ≈
            -(30.0 - 35.0) / 64 + sum(x -> x - 30.0, [28.0, 31.0, 29.0]) / 9
    end

    @testset "reevaluation resets recorded sites" begin
        vi = VarInfo(Xoshiro(6), branching(true))
        @test length(vi) == 2
        _, vi = init!!(Xoshiro(7), branching(false), vi, InitFromPrior())
        check_recorded(vi, [@varname(usual_minutes)])
        _, vi = init!!(Xoshiro(8), observation_only(), vi, InitFromPrior())
        check_recorded(vi, VarName[])
    end
end

open("results.toml", "w") do io
    TOML.print(io, RESULTS)
end
