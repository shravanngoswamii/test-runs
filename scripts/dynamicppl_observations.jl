using DynamicPPL, Distributions, ForwardDiff, Random, Test, TOML
using Logging: Warn
using OffsetArrays: OffsetArray

const fixed = parse(Bool, ENV["REVIEW_HEAD"])
const has_nans = DynamicPPL.DebugUtils._has_nans

struct NaNObservation end
Base.isnan(::NaNObservation) = true

struct NamedChoice{T,X} <: DiscreteUnivariateDistribution
    p::T
    right::X
end
Distributions.logpdf(d::NamedChoice, x) = x == d.right ? log(d.p) : log1p(-d.p)
Distributions.loglikelihood(d::NamedChoice, x) = logpdf(d, x)

@model function choices(observed, right)
    p ~ Beta(2, 2)
    for i in eachindex(observed)
        observed[i] ~ NamedChoice(p, right)
    end
    return p
end

@testset "NaN detection" begin
    @test has_nans(NaNObservation())
    for x in (missing, 0, true, 1.0f0, big"1.0", 1 + 2im, Inf, -Inf,
              ForwardDiff.Dual(1.0, 1.0))
        @test !has_nans(x)
    end
    for x in (NaN, NaN32, BigFloat(NaN), complex(1, NaN),
              ForwardDiff.Dual(NaN, 1.0))
        @test has_nans(x)
        @test has_nans((a=0, b=[missing, x]))
    end
    for x in (:right, "right", 'r', nothing, (choice=:right, data=["left", missing, 1.0]),
              Any[:right, [1.0]], OffsetArray([:left, :right], 0:1))
        if fixed
            @test !has_nans(x)
        else
            @test_throws MethodError has_nans(x)
        end
    end
    mixed = (choice=:right, data=["left", missing, NaN])
    if fixed
        @test has_nans(mixed)
    else
        @test_throws MethodError has_nans(mixed)
    end
end

results = Dict{String,Any}()
@testset "Public model workflows" begin
    for (label, data, right) in (("symbols", [:right, :left, :right], :right),
                               ("strings", ["right", "left", "right"], "right"))
        model = choices(data, right)
        if fixed
            @test_logs min_level=Warn @test check_model(Xoshiro(1), model)
            @test_logs min_level=Warn @test check_model(
                Xoshiro(1), model; error_on_failure=true, fail_if_discrete=true
            )
        else
            @test_throws MethodError check_model(Xoshiro(1), model)
        end
        @test logjoint(model, (; p=0.6)) ≈ logpdf(Beta(2, 2), 0.6) + 2log(0.6) + log(0.4)
        @test logprior(model, (; p=0.6)) ≈ logpdf(Beta(2, 2), 0.6)
        @test loglikelihood(model, (; p=0.6)) ≈ 2log(0.6) + log(0.4)
        @test returned(model, (; p=0.6)) == 0.6
        gradient = ForwardDiff.derivative(p -> logjoint(model, (; p)), 0.4)
        @test gradient ≈ 3 / 0.4 - 2 / 0.6
        draws = [rand(Xoshiro(i), model)[@varname(p)] for i in 1:5]
        vi = VarInfo(Xoshiro(11), model)
        @test collect(keys(vi)) == [@varname(p)]
        @test length(vi[:]) == 1
        results[label] = Dict(
            "draws" => draws, "varnames" => string.(collect(keys(vi))),
            "storage" => vi[:], "logjoint" => logjoint(model, (; p=0.6)),
            "stored_logjoint" => getlogjoint(vi), "gradient" => gradient,
        )
    end

    @model function numeric_observation(x)
        p ~ Normal()
        x ~ Normal(p)
    end
    @test_logs (:warn, r"Encountered a NaN value") @test !check_model(
        Xoshiro(1), numeric_observation(NaN)
    )
    @test_logs (:warn, r"Encountered a NaN value") @test_throws ErrorException check_model(
        Xoshiro(1), numeric_observation(NaN); error_on_failure=true
    )
end

open("results.toml", "w") do io
    TOML.print(io, results)
end
println("OBSERVATION_PROBES_PASSED fixed=", fixed)
