using AdvancedVI, Distributions, DynamicPPL, Test
using LogDensityProblems: logdensity

@model function minibatch(y; datapoints=eachindex(y))
    location ~ Normal()
    for i in datapoints
        y[i] ~ Normal(location, 1.0)
    end
end

model = minibatch([1.0, 2.0, 3.0])
ext = Base.get_extension(AdvancedVI, :AdvancedVIDynamicPPLExt)
target = ext.DynamicPPLModelLogDensityFunction(model, VarInfo(model); use_hessian=false)
params = [0.5]
prior = logpdf(Normal(), 0.5)
likelihood = Normal(0.5, 1.0)
@testset "Full batch and minibatch" begin
    @test logdensity(target, params) ≈ prior + sum(logpdf.(likelihood, [1.0, 2.0, 3.0]))
    AdvancedVI.subsample(target, [1, 3])
    if only(ARGS) == "base"
        @test logdensity(target, params) ≈ prior + 1.5sum(logpdf.(likelihood, [1.0, 3.0]))
    else
        @test_throws KeyError logdensity(target, params)
    end
end
