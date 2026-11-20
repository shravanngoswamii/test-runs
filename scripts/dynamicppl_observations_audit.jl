using DynamicPPL, Distributions, Random, Test

const fixed = parse(Bool, ENV["REVIEW_HEAD"])
const has_nans = DynamicPPL.DebugUtils._has_nans

struct CountingNaN
    calls::Base.RefValue{Int}
    result::Bool
end
Base.isnan(x::CountingNaN) = (x.calls[] += 1; x.result)

struct ThrowingNaN end
Base.isnan(::ThrowingNaN) = error("custom isnan failed")

struct MissingNaN end
Base.isnan(::MissingNaN) = missing

@show VERSION pkgversion(DynamicPPL) pkgversion(Distributions)

@testset "Dispatch and error preservation" begin
    @test applicable === Core.applicable
    for result in (false, true)
        x = CountingNaN(Ref(0), result)
        @test applicable(isnan, x)
        @test x.calls[] == 0
        @test has_nans(x) === result
        @test x.calls[] == 1
    end
    @test_throws ErrorException has_nans(ThrowingNaN())
    @test ismissing(has_nans(MissingNaN()))
    @test isnan(missing) === missing
    @test has_nans(missing) === false
    @test (@inferred has_nans(1.0)) === false
    @test (@inferred has_nans(NaN)) === true
end

@testset "Recursion boundary" begin
    @test has_nans((a=[1.0, NaN],))
    for x in ((NaN,), Dict(:x => NaN), (x=(NaN,),))
        if fixed
            @test !has_nans(x)
        else
            @test_throws MethodError has_nans(x)
        end
    end
end

example_result = try
    include("dynamicppl_observations_example.jl")
catch err
    err isa LoadError ? err.error : err
end

@testset "PR example API contract" begin
    if fixed
        @test example_result === true
    else
        @test example_result isa MethodError
        @test example_result.f === isnan
        @test example_result.args == (:right,)
    end
    d = SymbolChoice()
    @test loglikelihood(d, :right) == -log(2)
    @test eltype(d) === Symbol
    @test_throws MethodError logpdf(d, :right)
    @test_throws MethodError rand(Xoshiro(1), d)
    @test logjoint(choice(:right), (; x=0.0)) ≈ logpdf(Normal(), 0.0) - log(2)
    @test_throws MethodError logpdf(Dirac(:right), :right)
    @test_throws MethodError loglikelihood(Dirac(:right), :right)
    @test_throws MethodError DiscreteNonParametric([:left, :right], [0.5, 0.5])
end

println("AUDIT_PASSED fixed=", fixed)
