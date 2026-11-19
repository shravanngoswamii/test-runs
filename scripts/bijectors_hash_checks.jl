using Bijectors, Distributions, Test, TOML
using Bijectors: Shift, Scale

const FIXED = get(ENV, "REVIEW_HEAD", "false") == "true"
const VB = Bijectors.VectorBijectors

source = read(joinpath(@__DIR__, "..", "head-tests", "test", "bijectors", "equality.jl"), String)
added = match(r"@testset \"Hashing\" begin.*\z"s, source).match
failure = try
    include_string(@__MODULE__, added, "new-hash-regressions.jl")
    nothing
catch err
    while err isa LoadError
        err = err.error
    end
    err
end

@testset "New assertions distinguish the fix" begin
    if FIXED
        @test failure === nothing
    else
        @test failure isa Test.TestSetException
        if failure isa Test.TestSetException
            @test failure.fail == 21
            @test failure.error == 0
        end
    end
end

violations = 0
@testset "Hash contracts" begin
    arrays = ([1], [1.0], Float32[1], BigFloat[1], [true], view([1.0], :))
    for a in arrays, b in arrays, constructor in (Shift, Scale, x -> Bijectors.Inverse(Shift(x)))
        first, second = constructor(a), constructor(b)
        @test isequal(first, second)
        valid = all(h -> hash(first, h) == hash(second, h), UInt[0, 1, typemax(UInt)]) &&
            length(Set(Any[first, second])) == 1 && get(Dict{Any,Int}(first => 1), second, 0) == 1
        if FIXED
            @test valid
        else
            global violations += !valid
        end
    end
    distributions = (
        product_distribution([Normal(), Beta(2, 2)]),
        product_distribution((a=Normal(), b=Beta(2, 2))),
        product_distribution(fill(Beta(2, 2), 3)),
        reshape(product_distribution([Normal(), Beta(2, 2)]), (1, 2)),
        Dirichlet([1.0, 2.0, 3.0]),
    )
    for d in distributions, f in (VB.to_vec, VB.from_vec, VB.to_linked_vec, VB.from_linked_vec)
        first = f(d)
        second = deepcopy(first)
        if isequal(first, second)
            valid = all(h -> hash(first, h) == hash(second, h), UInt[0, 1, typemax(UInt)])
            if FIXED
                @test valid
                @test get(Dict{Any,Int}(first => 1), second, 0) == 1
            else
                global violations += !valid
            end
        end
    end
end

open("checks.toml", "w") do io
    TOML.print(io, Dict("julia" => string(VERSION), "fixed" => FIXED, "base_violations" => violations))
end
