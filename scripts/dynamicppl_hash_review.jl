using AbstractMCMC: SamplingOutput
using BangBang: setindex!!, empty!!
using Bijectors
using Distributions
using DynamicPPL
using DynamicPPL.VarNamedTuples: PartialArray, GrowableArray
using ForwardDiff
using LogDensityProblems: logdensity
using Random: Xoshiro
using Serialization
using Test
using TOML

const DP = DynamicPPL
const HEAD = get(ENV, "REVIEW_HEAD", "false") == "true"
const SEEDS = UInt[0, 1, 0x01234567, typemax(UInt)]
const VIOLATIONS = Dict{String,Any}[]
const ERRORS = Dict{String,Any}[]
const COUNTS = Dict("equal_pairs" => 0, "unequal_pairs" => 0)
const WORKFLOW = Dict{String,Any}()

function checkpair(label, a, b; require_equal=true, required=true)
    equal = try
        isequal(a, b)
    catch err
        push!(ERRORS, Dict("label" => label, "operation" => "isequal", "error" => sprint(showerror, err)))
        required && @test false
        return
    end
    require_equal && @test equal
    COUNTS[equal ? "equal_pairs" : "unequal_pairs"] += 1
    try
        hashes_agree = all(h -> hash(a, h) == hash(b, h), SEEDS)
        unique_count = length(Set(Any[a, b]))
        dictionary = Dict{Any,Int}(a => 1)
        found = get(dictionary, b, 0)
        dictionary[b] = 2
        expected_count = equal ? 1 : 2
        valid = (!equal || hashes_agree) && unique_count == expected_count &&
            length(dictionary) == expected_count && (!equal || found == 1)
        if !valid
            push!(VIOLATIONS, Dict(
                "label" => label, "left_type" => string(typeof(a)),
                "right_type" => string(typeof(b)), "equal" => equal,
                "hashes_agree" => hashes_agree, "set_length" => unique_count,
                "dict_length" => length(dictionary), "lookup" => found,
            ))
        end
        if required && HEAD
            @test valid
            if equal
                @test dictionary[a] == 2
                @test b in Set(Any[a])
                delete!(dictionary, a)
                @test isempty(dictionary)
            end
        elseif !equal
            @test unique_count == 2
            @test length(dictionary) == 2
        end
    catch err
        err isa Test.TestSetException && rethrow()
        push!(ERRORS, Dict("label" => label, "operation" => "hash or Dict", "error" => sprint(showerror, err)))
        required && @test false
    end
end

@model function commute(recorded_minutes)
    usual_minutes ~ Normal(35, 8)
    variability ~ LogNormal(1, 0.25)
    for day in eachindex(recorded_minutes)
        recorded_minutes[day] ~ Normal(usual_minutes, variability)
    end
    return usual_minutes, variability
end

failure = try
@testset "Hash contract review" begin
    @testset "arrays and numeric representations" begin
        values = Any[[1], [1.0], Float32[1], BigFloat[1], Rational{Int}[1], Bool[true], GrowableArray([1]), view([1.0], :)]
        for a in values, b in values
            checkpair("growable/plain $(typeof(a)) $(typeof(b))", GrowableArray(collect(a)), b)
            checkpair("FixedTransform $(typeof(a)) $(typeof(b))", FixedTransform(a), FixedTransform(b))
            for transform in (NoTransform(), Unlink(), DynamicLink(), FixedTransform(identity))
                checkpair("TransformedValue $(typeof(a)) $(typeof(b)) $transform", TransformedValue(a, transform), TransformedValue(b, transform))
            end
            checkpair("ParamsWithStats arrays", ParamsWithStats(VarNamedTuple(; travel=a), (; density=[1])), ParamsWithStats(VarNamedTuple(; travel=b), (; density=[1.0])))
        end
        scalars = Any[0, 0.0, -0.0, Float32(0), BigFloat(0), 1, 1.0, 1f0, big(1), 1//1, true, false, NaN, Float32(NaN), Inf, -Inf]
        for acc in (LogPriorAccumulator, LogLikelihoodAccumulator, LogJacobianAccumulator)
            for a in scalars, b in scalars
                checkpair("$acc numeric $a $b", acc(a), acc(b); require_equal=false)
            end
        end
        for a in (NaN, -0.0, 0.0, missing, Inf)
            checkpair("special transformed value $a", TransformedValue([a], NoTransform()), TransformedValue([a], NoTransform()))
        end
        checkpair("transform role differs", TransformedValue([1.0], NoTransform()), TransformedValue([1.0], DynamicLink()); require_equal=false)
        checkpair("accumulator role differs", LogPriorAccumulator(1), LogLikelihoodAccumulator(1); require_equal=false)
        checkpair("stats differ", ParamsWithStats(VarNamedTuple(; x=[1]), (; lp=1.0)), ParamsWithStats(VarNamedTuple(; x=[1]), (; lp=2.0)); require_equal=false)
    end

    @testset "partial arrays and masks" begin
        for dims in ((), (0,), (3,), (2, 2), (2, 1, 2), (0, 2))
            n = prod(dims)
            data = reshape(collect(1:n), dims)
            for bits in 0:(2^n - 1)
                mask = reshape([!iszero(bits & (1 << (i-1))) for i in 1:n], dims)
                reference = PartialArray(copy(data), copy(mask))
                variants = (
                    PartialArray(reshape(Float64.(vec(data)), dims), copy(mask)),
                    PartialArray(GrowableArray(copy(data)), copy(mask)),
                    PartialArray(copy(data), GrowableArray(copy(mask))),
                    PartialArray(GrowableArray(copy(data)), GrowableArray(copy(mask))),
                    PartialArray(copy(data), BitArray(mask)),
                    PartialArray(view(copy(data), ntuple(_ -> Colon(), length(dims))...), copy(mask)),
                )
                for (i, other) in enumerate(variants)
                    checkpair("partial $dims mask $bits storage $i", reference, other)
                    checkpair("nested VNT partial $dims mask $bits storage $i", VarNamedTuple(; travel=reference), VarNamedTuple(; travel=other))
                end
            end
        end
        a = PartialArray([1, 100], [true, false])
        b = PartialArray([1.0, -999.0], [true, false])
        checkpair("masked entries ignored", a, b)
        unknown = Vector{String}(undef, 3)
        unknown[2] = "recorded"
        other = Vector{Any}(undef, 3)
        other[2] = "recorded"
        checkpair("unassigned references are not read", PartialArray(unknown, [false, true, false]), PartialArray(other, [false, true, false]))
        checkpair("different masks", PartialArray([1, 1], [true, false]), PartialArray([1, 1], [false, true]); require_equal=false)
        @test !isequal(VarNamedTuple(; a=1, b=2), VarNamedTuple(; b=2, a=1))
        for dims in (1, 2, 3)
            template = zeros(Int, ntuple(_ -> 1, dims))
            vn = dims == 1 ? @varname(travel[1]) : dims == 2 ? @varname(travel[1,1]) : @varname(travel[1,1,1])
            templated = DP.templated_setindex!!(VarNamedTuple(), 1, vn, template)
            growable = setindex!!(VarNamedTuple(), 1.0, vn)
            checkpair("public VNT insertion dimensions $dims", templated, growable)
        end
    end

    @testset "transform strategies" begin
        names = [@varname(travel), @varname(scale), @varname(travel[1])]
        for strategy in (LinkSome, UnlinkSome)
            fallback = strategy === LinkSome ? UnlinkAll() : LinkAll()
            a = strategy(Set(names), fallback)
            b = strategy(Set(reverse(names)), deepcopy(fallback))
            checkpair("$strategy set order", a, b)
            checkpair("$strategy copy", a, deepcopy(a))
        end
        a = WithTransforms(VarNamedTuple(; travel=FixedTransform([1])), UnlinkAll())
        b = WithTransforms(VarNamedTuple(; travel=FixedTransform([1.0])), UnlinkAll())
        checkpair("WithTransforms nested fixed arrays", a, b)
        nested1 = LinkSome(Set([@varname(travel)]), UnlinkSome(Set([@varname(scale)]), LinkAll()))
        checkpair("nested fallback", nested1, deepcopy(nested1))
        for transform in (Bijectors.Shift([1.0, 2.0]), Bijectors.Scale([2.0, 3.0]))
            checkpair("dependency bijector $(typeof(transform))", FixedTransform(transform), FixedTransform(deepcopy(transform)); require_equal=false, required=false)
        end
    end

    @testset "model evaluation and dictionary keys" begin
        model = commute([28.0, 31.0, 29.0])
        params = (; usual_minutes=30.0, variability=2.0)
        for transform in (UnlinkAll(), LinkAll())
            a = VarInfo(Xoshiro(1), model, InitFromParams(params, nothing), transform)
            b = VarInfo(Xoshiro(1), model, InitFromParams(deepcopy(params), nothing), transform)
            checkpair("model VarInfo $transform", a, b)
            checkpair("vector values $transform", get_vector_values(a), get_vector_values(b))
            for vn in keys(a)
                checkpair("model transformed value $vn $transform", DP.get_transformed_value(a, vn), DP.get_transformed_value(b, vn))
            end
            original = deepcopy(get_vector_values(a))
            for _ in 1:3
                hash(a)
            end
            @test isequal(get_vector_values(a), original)
            fixed1 = DP.get_fixed_transforms(model, transform)
            fixed2 = DP.get_fixed_transforms(model, transform)
            checkpair("real fixed transforms $transform", fixed1, fixed2; require_equal=false, required=false)
        end
        first = ParamsWithStats(InitFromParams(params, nothing), model)
        second = ParamsWithStats(InitFromParams(deepcopy(params), nothing), model)
        checkpair("real ParamsWithStats", first, second)
        buffer = IOBuffer()
        serialize(buffer, first)
        seekstart(buffer)
        checkpair("serialized ParamsWithStats", first, deserialize(buffer))
        samples = SamplingOutput(reshape([first, second], 2, 1))
        predictions = predict(Xoshiro(2), commute(fill(missing, 2)), samples)
        @test length(predictions.samples) == 2
        @test predictions.samples == predict(Xoshiro(2), commute(fill(missing, 2)), samples).samples
        WORKFLOW["logjoint"] = first.stats.logjoint
        WORKFLOW["predictions"] = [ps.params[@varname(recorded_minutes[1])] for ps in vec(predictions.samples)]
        WORKFLOW["gradient"] = ForwardDiff.gradient(x -> logjoint(model, (; usual_minutes=x[1], variability=x[2])), [30.0, 2.0])
        ldf = LogDensityFunction(model, getlogjoint_internal, UnlinkAll())
        @test logdensity(ldf, [30.0, 2.0]) ≈ first.stats.logjoint
        dictionary = Dict{Any,Int}()
        for i in 1:100
            dictionary[TransformedValue([float(i)], NoTransform())] = i
        end
        if HEAD
            @test all(i -> dictionary[TransformedValue([float(i)], NoTransform())] == i, 1:100)
        end
    end
end
    nothing
catch err
    err
end

open("results.toml", "w") do io
    TOML.print(io, Dict(
        "julia" => string(VERSION), "head" => HEAD, "counts" => COUNTS,
        "violations" => VIOLATIONS, "errors" => ERRORS, "workflow" => WORKFLOW,
    ))
end
println("Hash violations: ", length(VIOLATIONS), ", unexpected operations: ", length(ERRORS))
failure === nothing || throw(failure)
