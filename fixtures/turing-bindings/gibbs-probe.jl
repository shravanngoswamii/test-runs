using AbstractMCMC: SamplingOutput
using DynamicPPL, Random, StableRNGs, Turing
using Turing.Inference: Particle, advance!, particle_rng
using Test: @testset

if only(ARGS) == "mark-tuple-map"
    @eval Turing.Inference begin
        Libtask.@might_produce(DynamicPPL.map_accumulators!!)
        Libtask.might_produce(::Type{<:Tuple{typeof(map),Function,DynamicPPL.AccumulatorTuple}}) = true
    end
elseif only(ARGS) == "keep-evaluator-call"
    @eval Turing.Inference begin
        @noinline run_particle_model(model, args...; kwargs...) = model.f(model, args...; kwargs...)
        function evaluate_particle!(model::DynamicPPL.Model, args...; kwargs...)
            _, vi = run_particle_model(model, args...; kwargs...)
            Libtask.get_taped_globals(Particle).varinfo = vi
            return nothing
        end
    end
end

@model function dynamic_bernoulli_normal(y_obs=2.0)
    b ~ Bernoulli(0.3)
    θ = zeros(2)
    if b == 0
        θ[1] ~ Normal()
        y_obs ~ Normal(θ[1], 0.5)
    else
        θ[1] ~ Normal()
        θ[2] ~ Normal()
        y_obs ~ Normal(θ[1] + θ[2], 0.5)
    end
end
function diagnose(model, seed)
    particle = Particle(model, particle_rng(Xoshiro(seed)))
    scores = Float64[]
    while (score = advance!(particle)) !== nothing
        push!(scores, score)
    end
    vi = particle.varinfo
    vals = DynamicPPL.get_parameter_values(vi)
    @show seed scores keys(vals) DynamicPPL.getloglikelihood(vi) loglikelihood(model, vals)
    flush(stdout)
end

model = dynamic_bernoulli_normal(2.0)
@show VERSION pathof(DynamicPPL) pathof(Turing) isdefined(DynamicPPL, :BindingTable)
for seed in 1:5
    diagnose(model, seed)
end

function compare_draws(model)
    for sampler in (
        Gibbs((@varname(b), @varname(θ)) => PG(50)),
        Gibbs((@varname(b), @varname(θ)) => PG(20), @varname(θ) => MH()),
    )
        kwargs = (; discard_initial=1_000, progress=false, verbose=false)
        for chain_type in (SamplingOutput, VNChain)
            out = sample(StableRNG(42), model, sampler, 2_000; kwargs..., chain_type)
            bs = chain_type === SamplingOutput ?
                [d.params[@varname(b)] for d in out.samples] : out[@varname(b)]
            @show sampler chain_type count(==(1), bs) / length(bs)
            flush(stdout)
        end
    end
end

@testset "Gibbs in test scope" begin
    @model function dynamic_bernoulli_normal_local(y_obs=2.0)
        b ~ Bernoulli(0.3)
        θ = zeros(2)
        if b == 0
            θ[1] ~ Normal(0.0, 1.0)
            y_obs ~ Normal(θ[1], 0.5)
        else
            θ[1] ~ Normal(0.0, 1.0)
            θ[2] ~ Normal(0.0, 1.0)
            y_obs ~ Normal(θ[1] + θ[2], 0.5)
        end
    end
    model = dynamic_bernoulli_normal_local(2.0)
    for seed in 1:5
        diagnose(model, seed)
    end
    compare_draws(model)
end
