using DynamicPPL
using DynamicPPL.VarNamedTuples: GrowableArray, templated_setindex!!
using BangBang: setindex!!
using Test

root = joinpath(@__DIR__, "..", "head-tests", "test")
acc = read(joinpath(root, "accumulators.jl"), String)
chains = read(joinpath(root, "chains.jl"), String)
transforms = read(joinpath(root, "transformed_values.jl"), String)
vnt = read(joinpath(root, "varnamedtuple.jl"), String)
snippets = [
    match(r"@test hash\(LogPriorAccumulator\(1\)\).*", acc).match,
    match(r"@testset \"ParamsWithStats hash\" begin.*?\nend"s, chains).match,
    match(r"@testset \"hash consistent with isequal\" begin.*?\nend"s, transforms).match,
    match(r"        vnt_templated = templated_setindex!!\(.*?@test hash\(GrowableArray\(\[1\]\)\) == hash\(\[1\]\)"s, vnt).match,
]

failure = try
    @testset "Exact added PR assertions" begin
        for (i, source) in enumerate(snippets)
            include_string(@__MODULE__, source, "added-hash-tests-$i.jl")
        end
    end
    nothing
catch err
    while err isa LoadError
        err = err.error
    end
    err
end

@testset "PR test discrimination" begin
    if get(ENV, "REVIEW_HEAD", "false") == "true"
        @test failure === nothing
    else
        @test failure isa Test.TestSetException
        if failure isa Test.TestSetException
            @test failure.fail == 9
            @test failure.error == 0
        end
    end
end
