using AbstractPPL, DynamicPPL, Test
using DynamicPPL.VarNamedTuples: templated_setindex!!

source = read(joinpath(@__DIR__, "..", "head-tests", "test", "varnamedtuple.jl"), String)
added = match(r"@testset \"binding templates from storage\".*?(?=\n@info \"Completed)"s, source).match
failure = try
    include_string(@__MODULE__, added, "added-storage-tests.jl")
    nothing
catch err
    while err isa LoadError
        err = err.error
    end
    err
end
@testset "Added assertions distinguish the PR" begin
    if get(ENV, "REVIEW_HEAD", "false") == "true"
        @test failure === nothing
    else
        @test failure isa Test.TestSetException
        if failure isa Test.TestSetException
            @test failure.fail + failure.error == 3
        end
    end
end
include(joinpath(@__DIR__, "..", "target", "test", "varnamedtuple.jl"))
if get(ENV, "REVIEW_HEAD", "false") == "true"
    using Documenter, Distributions
    Documenter.DocMeta.setdocmeta!(DynamicPPL, :DocTestSetup, :(using DynamicPPL, Distributions); recursive=true)
    Documenter.doctest(DynamicPPL; manual=false, doctestfilters=[r"└ @ .+:[0-9]+"])
end
