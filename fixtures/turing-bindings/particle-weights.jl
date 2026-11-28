using DynamicPPL, Random, StableRNGs, Test, Turing
using Turing.Inference: Particle, advance!, particle_rng

@testset "Particle weights in a local model" begin
    @model function branching(; reading=2.0)
        b ~ Bernoulli(0.3)
        θ = zeros(2)
        θ[1] ~ Normal()
        if b == 0
            reading ~ Normal(θ[1], 0.5)
        else
            θ[2] ~ Normal()
            reading ~ Normal(θ[1] + θ[2], 0.5)
        end
    end

    model = branching(; reading=2.0)
    branches = Set{Int}()
    for seed in 1:5
        particle = Particle(model, particle_rng(Xoshiro(seed)))
        score = advance!(particle)
        @test advance!(particle) === nothing
        values = DynamicPPL.get_parameter_values(particle.varinfo)
        @test score == loglikelihood(model, values)
        push!(branches, values[@varname(b)])
    end
    @test branches == Set((0, 1))

    sampler = Gibbs((@varname(b), @varname(θ)) => PG(50))
    chain = sample(
        StableRNG(42), model, sampler, 2_000; discard_initial=1_000, progress=false,
    )
    bs = chain[@varname(b)]
    @test count(==(1), bs) / length(bs) ≈ 0.394 atol = 0.05
end
