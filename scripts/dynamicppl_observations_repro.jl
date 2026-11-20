using Pkg
if !parse(Bool, ENV["REVIEW_HEAD"])
    Pkg.add([PackageSpec(name="Turing", version=v"0.49.0"),
             PackageSpec(name="Distributions", version=v"0.25.134")])
    Pkg.update()
end

using DynamicPPL, Distributions, Random, Test, TOML

struct SymbolChoice <: Distribution{Univariate,ValueSupport}
    probability_right::Float64
end
Distributions.logpdf(d::SymbolChoice, x::Symbol) =
    x === :right ? log(d.probability_right) : log1p(-d.probability_right)
Distributions.loglikelihood(d::SymbolChoice, x::Symbol) = logpdf(d, x)
Base.rand(rng::AbstractRNG, d::SymbolChoice) =
    rand(rng) < d.probability_right ? :right : :left

@model function choices(observed)
    p ~ Beta(2, 2)
    for i in eachindex(observed)
        observed[i] ~ SymbolChoice(p)
    end
end

model = choices([:right, :left, :right])
@testset "Reported failure before the fix" begin
    @test logjoint(model, (; p=0.6)) ≈ logpdf(Beta(2, 2), 0.6) + 2log(0.6) + log(0.4)
    err = try
        check_model(Xoshiro(1), model)
    catch e
        e
    end
    @show err
    @test err isa MethodError && err.f === isnan && err.args == (:right,)
    @test_throws MethodError DynamicPPL.DebugUtils._has_nans(:right)
    @test DynamicPPL.DebugUtils._has_nans((x=1.0, y=[NaN, 0.5]))
    @test !DynamicPPL.DebugUtils._has_nans((x=1.0, y=[missing, 0.5]))
end

if !parse(Bool, ENV["REVIEW_HEAD"])
    @eval using Turing
    @testset "Exact reported Turing workflow" begin
        @test size(sample(Xoshiro(1), model, Prior(), 5; progress=false, check_model=false), 1) == 5
        err = try
            sample(Xoshiro(1), model, Prior(), 5; progress=false)
        catch e
            e
        end
        @show err
        @test err isa MethodError && err.f === isnan && err.args == (:right,)
    end
end

Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
println("REPRODUCED_ISNAN_SYMBOL_ON_", pkgversion(DynamicPPL))
