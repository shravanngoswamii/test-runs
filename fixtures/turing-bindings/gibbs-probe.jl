using AbstractMCMC: SamplingOutput
using DynamicPPL, Random, StableRNGs, Turing
using Turing.Inference: Particle, advance!, particle_rng

if only(ARGS) == "mark-map"
    @eval Turing.Inference Libtask.@might_produce(DynamicPPL.map_accumulators!!)
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

sampler = Gibbs((@varname(b), @varname(θ)) => PG(50))
out = sample(
    StableRNG(42), model, sampler, 2_000; discard_initial=1_000,
    progress=false, verbose=false, chain_type=SamplingOutput,
)
println("Gibbs PG frequency = ", count(d -> d.params[@varname(b)] == 1, out.samples) / length(out.samples))
