using AbstractMCMC: SamplingOutput
using DynamicPPL, MCMCChains, Random, Test, Turing

@model function commute(observed)
    typical ~ Normal(30, 5)
    observed ~ Normal(typical, 4)
    twice := 2typical
    return observed
end

@model function explicit()
    typical ~ Normal(30, 5)
    observed ~ Normal(typical, 4)
    return observed
end

before = only(ARGS) == "before"
model = commute(32.0)
@testset "Particle output" begin
    for sampler in (SMC(), PG(5)), chain_type in (SamplingOutput, MCMCChains.Chains)
        chain = sample(
            Xoshiro(51), model, sampler, 5; progress=false, verbose=false, chain_type
        )
        if chain isa SamplingOutput
            if before
                @test all(d -> d.stats.loglikelihood == 0, chain.samples)
                @test all(d -> !haskey(d.params, @varname(twice)), chain.samples)
            else
                for d in chain.samples
                    typical = d.params[@varname(typical)]
                    @test d.stats.loglikelihood ≈ logpdf(Normal(typical, 4), 32.0)
                    @test d.params[@varname(twice)] == 2typical
                end
            end
        else
            if before
                @test all(iszero, chain[:loglikelihood])
                @test :twice ∉ MCMCChains.names(chain)
            else
                @test chain[:loglikelihood] ≈ logpdf.(Normal.(chain[:typical], 4), 32.0)
                @test chain[:twice] == 2 .* chain[:typical]
            end
        end
    end
end

@testset "An explicit observation reaches GibbsConditional" begin
    conditioned = condition(explicit(); observed=32.0)
    sampler = Gibbs(:typical => GibbsConditional(v -> Normal(v[@varname(observed)], 1)))
    if before
        @test_throws KeyError sample(Xoshiro(51), conditioned, sampler, 5; progress=false)
    else
        out = sample(
            Xoshiro(51), conditioned, sampler, 5;
            progress=false, verbose=false, chain_type=SamplingOutput,
        )
        @test size(out) == (5, 1)
        new_model = decondition(conditioned, @varname(observed))
        predictions = predict(Xoshiro(52), new_model, out)
        @test all(d -> haskey(d.params, @varname(observed)), predictions.samples)
    end
end
