using ADTypes, Bijectors, DifferentiationInterface, ForwardDiff, Random, Test
using Enzyme: Enzyme, set_runtime_activity, Forward, Reverse, Const

include(joinpath(@__DIR__, "..", "target", "test", "test_resources.jl"))

ad_forward = AutoEnzyme(; mode=set_runtime_activity(Forward), function_annotation=Const)
ad_reverse = AutoEnzyme(; mode=set_runtime_activity(Reverse), function_annotation=Const)
c = only(filter(c -> c.tag === :reshaped_beta_special, generate_vector_testcases()))

failure = try
    Random.seed!(23)
    run_vector_case(c, [ad_forward, ad_reverse]; broken_adtypes=[ad_reverse])
    nothing
catch err
    err
end
@testset "Existing broken assertion now passes" begin
    @test failure isa Test.TestSetException
    if failure isa Test.TestSetException
        @test failure.error == 1
        @test failure.fail == 0
    end
end

@testset "The AD case passes without the broken marker" begin
    Random.seed!(23)
    run_vector_case(c, [ad_forward, ad_reverse])
end
