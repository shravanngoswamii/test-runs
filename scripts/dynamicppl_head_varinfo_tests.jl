using Test

head = get(ENV, "EXPECT_RECORDED_QUERIES", "false") == "true"
failure = try
    include(joinpath(@__DIR__, "..", "head-tests", "test", "varinfo.jl"))
    nothing
catch err
    while err isa LoadError
        err = err.error
    end
    err
end

@testset "PR test discrimination" begin
    if head
        @test failure === nothing
    else
        @test failure isa Test.TestSetException
        if failure isa Test.TestSetException
            @test failure.fail == 12
            @test failure.error == 0
        end
    end
end
