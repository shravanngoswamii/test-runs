using Test
fixed = parse(Bool, ENV["REVIEW_HEAD"])
result = try
    include("dynamicppl_observations_example.jl")
catch err
    err
end
@show result
@testset "PR description example" begin
    if fixed
        @test result === true
    else
        @test result isa MethodError && result.f === isnan && result.args == (:right,)
    end
end
