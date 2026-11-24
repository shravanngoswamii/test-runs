using DynamicPPL, Distributions, Test
import AbstractPPL

source = read("target/test/logdensityfunction.jl", String)
regression = match(
    r"(@testset \"LogDensityAt deprecation shim\" begin[\s\S]*?\nend)\n\nend\s*$",
    source,
).captures[1]
result = try
    include_string(Main, regression, "existing deprecation test")
    nothing
catch err
    while err isa LoadError
        err = err.error
    end
    err
end
@testset "Deprecation test invocation" begin
    if Base.JLOptions().depwarn == 0
        @test result isa Test.TestSetException
        @test result.fail == 1
        @test result.error == 0
    else
        @test result === nothing
    end
end
println("Checked ", only(ARGS), " with depwarn=", Base.JLOptions().depwarn)
