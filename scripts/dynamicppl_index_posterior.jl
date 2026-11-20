using AbstractMCMC: SamplingOutput
using Distributions, DynamicPPL, Test, TOML
using DynamicPPL: predict
using MCMCChains: Chains
using Random: Xoshiro

const HEAD = get(ENV, "REVIEW_HEAD", "false") == "true"

@model function one_trip_slot()
    trip_times = zeros(3, 3)
    trip_times[4] ~ Normal()
    return trip_times[4]
end
@model function commute(observed)
    usual_times = zeros(2, 2)
    future_times = zeros(2, 2)
    for i in eachindex(observed)
        usual_times[i] ~ Normal(30, 5)
        observed[i] ~ Normal(usual_times[i], 2)
        future_times[i] ~ Normal(usual_times[i], 2)
    end
    return future_times
end

@testset "Linear parameters in matrix-shaped models" begin
    params = VarNamedTuple((@varname(trip_times[4]) => 1.0,))
    selected = predict(Xoshiro(1), one_trip_slot(), params)
    @test HEAD ? isempty(selected.params) : haskey(selected.params, @varname(trip_times[1, 2]))
end

@testset "Posterior prediction for four commute routes" begin
    observed = [29.0 31.0; 32.0 28.0]
    model = commute(observed)
    # Independent Normal priors and one Normal observation per route give exact posteriors.
    posterior_variance = inv(inv(25.0) + inv(4.0))
    posterior_means = posterior_variance .* (30 / 25 .+ observed ./ 4)
    posterior_rng = Xoshiro(23)
    draws = map(CartesianIndices((6, 2))) do _
        times = [rand(posterior_rng, Normal(posterior_means[i], sqrt(posterior_variance))) for i in 1:4]
        VarNamedTuple((@varname(usual_times[i]) => times[i] for i in 1:4))
    end
    posterior = SamplingOutput(draws; iterations=10:10:60)
    complete = predict(Xoshiro(8), model, posterior; include_all=true)
    selected = predict(Xoshiro(8), model, posterior)
    @test selected.iterations == posterior.iterations
    for i in eachindex(complete.samples)
        all_draw = complete.samples[i]
        predicted = selected.samples[i]
        usual = all_draw.params[@varname(usual_times)]
        future = all_draw.params[@varname(future_times)]
        @test predicted.params[@varname(future_times)] == future
        @test predicted.stats == all_draw.stats
        @test all_draw.stats.loglikelihood ≈ sum(logpdf(Normal(usual[j], 2), observed[j]) for j in 1:4)
        @test logjoint(model, all_draw.params) ≈ all_draw.stats.logjoint
        @test returned(model, all_draw.params) == future
        @test HEAD ? !haskey(predicted.params, @varname(usual_times[1, 2])) : haskey(predicted.params, @varname(usual_times[1, 2]))
    end
    chain = convert(Chains, posterior)
    prediction_chain = predict(Xoshiro(8), model, chain)
    @test size(prediction_chain) == (6, 4, 2)
    @test all(startswith(string(name), "future_times[") for name in names(prediction_chain))
end
open("results.toml", "w") do io
    TOML.print(io, Dict("julia" => string(VERSION), "head" => HEAD, "workflow" => "Exact Normal posterior draws, prediction, scoring, returned values and MCMCChains"))
end
