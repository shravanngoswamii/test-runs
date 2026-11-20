using Pkg, TOML
if ARGS[1] == "setup"
    Pkg.activate("release-env")
    Pkg.develop(path=abspath("release-target"))
    Pkg.add([PackageSpec(name="Turing", version=v"0.49.0"),
             PackageSpec(name="Distributions", version=v"0.25.134")])
    Pkg.compat("Turing", "=0.49.0")
    Pkg.compat("Distributions", "=0.25.134")
    Pkg.update()
    Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
    exit()
elseif ARGS[1] == "compare"
    @assert TOML.parsefile("release-before.toml") == TOML.parsefile("release-fixed.toml")
    println("TURING_SEEDED_VALUES_AND_DENSITIES_IDENTICAL")
    exit()
end

using Turing, DynamicPPL, Distributions, Random, Test

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

fixed = ARGS[1] == "fixed"
model = choices([:right, :left, :right])
parameter_values(chain) = [
    Turing.FlexiChains.parameters_at(chain; iter=i, chain=1)[@varname(p)]
    for i in 1:5
]
@testset "Reported Turing workflow" begin
    @test pkgversion(Turing) == v"0.49.0"
    @test pkgversion(DynamicPPL) == v"0.42.14"
    @test pkgversion(Distributions) == v"0.25.134"
    lp = logjoint(model, (; p=0.6))
    @test lp ≈ logpdf(Beta(2, 2), 0.6) + 2log(0.6) + log(0.4)
    unchecked = sample(Xoshiro(12), model, Prior(), 5; progress=false, check_model=false)
    draws = parameter_values(unchecked)
    if fixed
        @test check_model(model)
        checked = sample(Xoshiro(12), model, Prior(), 5; progress=false)
        @test size(checked) == size(unchecked) == (5, 1)
        @test parameter_values(checked) == draws
    else
        @test_throws MethodError check_model(model)
        @test_throws MethodError sample(Xoshiro(12), model, Prior(), 5; progress=false)
    end
    result = Dict("draws"=>draws, "logjoint"=>lp,
                  "draw_densities"=>[logjoint(model, (; p)) for p in draws])
    open("release-$(ARGS[1]).toml", "w") do io
        TOML.print(io, result)
    end
end
