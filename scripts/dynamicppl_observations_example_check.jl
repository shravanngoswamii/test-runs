using Test
fixed = parse(Bool, ENV["REVIEW_HEAD"])
result = try
    include("dynamicppl_observations_example.jl")
catch err
    err isa LoadError ? err.error : err
end
@show result
@testset "PR description example" begin
    @test eltype(SymbolChoice()) === Symbol
    if fixed
        @test result === true
    else
        @test result isa MethodError && result.f === isnan && result.args == (:right,)
    end
end
