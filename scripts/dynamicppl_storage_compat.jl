using Test
using AbstractPPL

println("Julia ", VERSION, ", AbstractPPL ", pkgversion(AbstractPPL))
failure = try
    @eval using DynamicPPL
    nothing
catch err
    showerror(stdout, err, catch_backtrace())
    println()
    err
end
expected = ENV["REVIEW_REVISION"] == "head" && pkgversion(AbstractPPL) < v"0.15.3"
@testset "Loading with an allowed AbstractPPL version" begin
    @test (failure !== nothing) == expected
    if failure !== nothing
        @test occursin("of", sprint(showerror, failure))
    else
        values = DynamicPPL.VarNamedTuple(; x=1.0)
        @test values[only(keys(values))] == 1.0
        if ENV["REVIEW_REVISION"] == "head"
            @test zero(AbstractPPL.of(DynamicPPL.VarNamedTuple(; y=zeros(Float32, 2)))) == (y=Float32[0, 0],)
        end
    end
end
