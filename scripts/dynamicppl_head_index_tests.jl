using Distributions, DynamicPPL, Random, Test

source = read(joinpath(@__DIR__, "..", "head-tests", "test", "model.jl"), String)
added = match(r"@model function linear_local\(\).*?(?=\n@info \"Completed)"s, source).match
failure = try
    include_string(@__MODULE__, added, "added-index-tests.jl")
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
            @test failure.fail == 4
            @test failure.error == 0
        end
    end
end
