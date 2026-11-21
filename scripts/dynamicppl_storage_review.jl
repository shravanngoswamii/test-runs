using AbstractPPL, BangBang, Distributions, DynamicPPL, ForwardDiff, Test
using DynamicPPL.VarNamedTuples: templated_setindex!!
using Random: Xoshiro
using SparseArrays: sparse
using StaticArrays: SVector

const HEAD = get(ENV, "REVIEW_HEAD", "false") == "true"
println("Julia ", VERSION, ", DynamicPPL ", pkgversion(DynamicPPL), ", AbstractPPL ", pkgversion(AbstractPPL))

@testset "Public model control" begin
    @model function commute()
        travel_times = zeros(3)
        for day in eachindex(travel_times)
            travel_times[day] ~ Normal(30, 5)
        end
        return sum(travel_times)
    end
    params = VarNamedTuple(; travel_times=[25.0, 30.0, 35.0])
    @test returned(commute(), params) == 90.0
    @test logjoint(commute(), params) ≈ sum(logpdf.(Normal(30, 5), [25.0, 30.0, 35.0]))
    draw = DynamicPPL.predict(Xoshiro(1), commute(), VarNamedTuple(); include_all=true)
    println("Sample keys: ", collect(keys(draw.params)))
    template = HEAD ? zero(AbstractPPL.of(draw.params)).travel_times : zeros(3)
    partial = templated_setindex!!(VarNamedTuple(), 31.0, @varname(travel_times[2]), template)
    result = DynamicPPL.predict(Xoshiro(2), commute(), partial; include_all=true)
    @test result.params[@varname(travel_times[2])] == 31.0
    @test returned(commute(), result.params) == sum(result.params[@varname(travel_times)])
    @test isfinite(logjoint(commute(), result.params))
end

@testset "New storage template API" begin
    if !HEAD
        @test_throws MethodError AbstractPPL.of(VarNamedTuple(; y=zeros(Float32, 2)))
    else
        @test zero(AbstractPPL.of(VarNamedTuple(; y=zeros(Float32, 2)))) == (y=Float32[0, 0],)
        @test AbstractPPL.of(VarNamedTuple()) == AbstractPPL.of(NamedTuple())
        for x in (Float16(2), 3.0f0, 4.0, big"5.0", 6)
            template = AbstractPPL.of(VarNamedTuple(; x))
            @test typeof(zero(template).x) === typeof(x)
            @test iszero(zero(template).x)
        end
        for T in (Float32, Float64, BigFloat, Int8, ComplexF64, Real), dims in ((), (0,), (3,), (2, 3), (2, 1, 3))
            array = zeros(T, dims)
            template = AbstractPPL.of(VarNamedTuple(; array))
            rebuilt = zero(template).array
            @test size(rebuilt) == dims
            @test eltype(rebuilt) === T
            @test rebuilt !== array
            @test all(iszero, rebuilt)
        end
        partial = templated_setindex!!(VarNamedTuple(), 99.0f0, @varname(x[2, 3]), zeros(Float32, 4, 5))
        template = AbstractPPL.of(VarNamedTuple(; a=partial, b=1.0))
        @test template == AbstractPPL.of((a=(x=zeros(Float32, 4, 5),), b=0.0))
        @test size(zero(template).a.x) == (4, 5)
        @test partial[@varname(x[2, 3])] == 99.0f0
        @test !haskey(partial, @varname(x[1, 1]))
        @test AbstractPPL.of(VarNamedTuple(; a=(x=zeros(Float32, 2),))) ==
            AbstractPPL.of(VarNamedTuple(; a=VarNamedTuple(; x=zeros(Float32, 2))))
        unassigned = Vector{Real}(undef, 3)
        unassigned[2] = 99.0
        @test eltype(zero(AbstractPPL.of(VarNamedTuple(; unassigned))).unassigned) === Real
        @test !isassigned(unassigned, 1)
        duals = [ForwardDiff.Dual(2.0, 1.0)]
        @test eltype(zero(AbstractPPL.of(VarNamedTuple(; duals))).duals) === eltype(duals)
        for value in (true, Int8(1), UInt(2), big(3), 1//2, ForwardDiff.Dual(1.0, 1.0))
            @test_throws ArgumentError AbstractPPL.of(VarNamedTuple(; a=VarNamedTuple(; x=value)))
        end
        for value in ("a", nothing, missing, (1.0, 2.0), view(zeros(2), :), SVector(1.0, 2.0), sparse([1.0, 0.0]), [[1.0], [2.0]])
            err = try
                AbstractPPL.of(VarNamedTuple(; a=VarNamedTuple(; x=value)))
            catch err
                err
            end
            @test err isa ArgumentError
            @test occursin("`a.x`", sprint(showerror, err))
        end
        growable = setindex!!(VarNamedTuple(), 1.0, @varname(x[2]))
        @test_throws ArgumentError AbstractPPL.of(growable)
        @test_throws ArgumentError AbstractPPL.of(VarNamedTuple(; a=growable))
        indexed = templated_setindex!!(VarNamedTuple(), 1.0, @varname(x[2].a), [(a=0.0,), (a=0.0,)])
        @test_throws ArgumentError AbstractPPL.of(indexed)
    end
end
