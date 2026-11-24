using DynamicPPL, Distributions, Random, Test, TOML, ForwardDiff
using Logging: with_logger, Warn
using OffsetArrays: OffsetArray

const mode = only(ARGS)
const fixed = mode == "head"
const original = mode == "base"

# The reviewer's reproducer, unchanged.
struct D <: DiscreteUnivariateDistribution end
Distributions.loglikelihood(::D, ::Tuple) = 0.0
@model m(y) = (x ~ Normal(); y ~ D())

@testset "Reviewer reproducer" begin
    logger = Test.TestLogger()
    if original
        @test_throws MethodError with_logger(logger) do
            check_model(m((:a, NaN)))
        end
    else
        result = with_logger(logger) do
            check_model(m((:a, NaN)))
        end
        @test result == !fixed
        @test count(r -> r.level == Warn && occursin("Encountered a NaN", string(r.message)),
                    logger.logs) == Int(fixed)
    end
end

source = read("head-debug-tests.jl", String)
fixture = match(r"struct NonNumericDist[^\n]+\nDistributions.loglikelihood[^\n]+\n", source).match
first_line = first(findfirst("    @testset \"non-numeric data\" begin", source))
next_test = first(findnext("    @testset \"incorrect use of condition\"", source, first_line))
regression = fixture * source[first_line:prevind(source, next_test)]
result = try
    include_string(Main, regression, "committed regression")
    nothing
catch err
    while err isa LoadError
        err = err.error
    end
    err
end
@testset "The committed tests distinguish each revision" begin
    if fixed
        @test result === nothing
    else
        @test result isa Test.TestSetException
        @test result.fail == 1
        @test result.error == Int(original)
    end
end

@model function observed(y)
    x ~ Normal()
    y ~ NonNumericDist()
    return x
end

@testset "Symbol observations and stored values" begin
    if original
        @test_throws MethodError check_model(Xoshiro(5), observed(:a))
    else
        @test check_model(Xoshiro(5), observed(:a))
    end
    vi = VarInfo(Xoshiro(12), observed(:a))
    results = Dict(
        "draws" => [rand(Xoshiro(i), observed(:a))[@varname(x)] for i in 1:3],
        "keys" => string.(collect(keys(vi))),
        "values" => vi[:],
        "density" => logjoint(observed(:a), (; x=0.4)),
        "returned" => returned(observed(:a), (; x=0.4)),
    )
    open("$mode-values.toml", "w") do io
        TOML.print(io, results; sorted=true)
    end
    if fixed
        @test results == TOML.parsefile("base-values.toml")
        @test results == TOML.parsefile("previous-values.toml")
    end
end

if fixed
    @testset "Numeric leaves and nested containers" begin
        for value in (NaN32, BigFloat(NaN), complex(1, NaN), ForwardDiff.Dual(NaN, 1.0))
            logger = Test.TestLogger()
            passed = with_logger(logger) do
                check_model(Xoshiro(5), observed((:a, (data=Any[missing, value],))))
            end
            @test !passed
            @test any(r -> occursin("Encountered a NaN", string(r.message)), logger.logs)
        end
        @test check_model(Xoshiro(5), observed((:a, (data=Any[missing, 1.0f0, big"1.0"],))))
        @test !DynamicPPL.DebugUtils._has_nans(OffsetArray([:a, :b], 0:1))
        @test DynamicPPL.DebugUtils._has_nans(OffsetArray(Any[:a, NaN], 0:1))
    end
end

struct CustomNaN end
Base.isnan(::CustomNaN) = true
@test DynamicPPL.DebugUtils._has_nans(CustomNaN()) == !fixed

module PRExample
    const result = try
        include(joinpath(@__DIR__, "dynamicppl_observations_example.jl"))
    catch err
        while err isa LoadError
            err = err.error
        end
        err
    end
end
@test original ? PRExample.result isa MethodError : PRExample.result === true
println("Observation review passed for ", mode)
