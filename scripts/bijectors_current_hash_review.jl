using Bijectors, Distributions, Test
using Bijectors: Shift, Scale, Logit, Inverse, TruncatedBijector, Permute, PlanarLayer,
    RadialLayer, InvertibleBatchNorm, Stacked, SimplexBijector, PDBijector

const VB = Bijectors.VectorBijectors
const HEAD = get(ENV, "REVIEW_HEAD", "false") == "true"
const SEEDS = (UInt(0), UInt(1), typemax(UInt))

function record(label, a, b; expected=true)
    eq = isequal(a, b)
    hashes = all(h -> hash(a, h) == hash(b, h), SEEDS)
    count = length(Set(Any[a, b]))
    found = get(Dict{Any,Int}(a => 1), b, 0)
    println(label, " => ", (; eq, hashes, count, found))
    if HEAD
        @test eq == expected
        @test count == (expected ? 1 : 2)
        @test found == Int(expected)
        expected && @test hashes
    end
end

println("Julia ", VERSION, " | Bijectors ", pkgversion(Bijectors), " | Distributions ",
    pkgversion(Distributions), " | ", pathof(Bijectors))
@testset "Public examples and changed types" begin
    d = product_distribution([Normal(), Beta()])
    example = hash(bijector(d)) == hash(bijector(d))
    println("PR example => ", example)
    @test example == HEAD
    record("Hong's bijector", bijector(d), bijector(d))
    record("Hong's inverse", inverse(bijector(d)), inverse(bijector(d)))
    @test length(Set([Shift([1.0, 2.0]), Shift([1.0, 2.0])])) == (HEAD ? 1 : 2)

    factories = (
        "Shift" => Shift, "Scale" => Scale, "Inverse" => (x -> Inverse(Shift(x))),
        "Logit" => (x -> Logit(x, [3.0, 4.0])),
        "TruncatedBijector" => (x -> TruncatedBijector(x, [3.0, 4.0])),
        "Permute" => (x -> Permute(reshape(x, 1, 2))),
        "PlanarLayer" => (x -> PlanarLayer(x, copy(x), 1)),
        "RadialLayer" => (x -> RadialLayer(x, copy(x), copy(x))),
        "InvertibleBatchNorm" => (x -> InvertibleBatchNorm(x, copy(x), copy(x), copy(x), 0.125, 0.25)),
        "Stacked" => (x -> Stacked([Shift(x)], [1:2])),
        "Elementwise" => (x -> VB.Elementwise(Shift(x), (2,))),
        "ProductVecTransform" => (x -> VB.ProductVecTransform([Shift(x)], [1:2], (2,))),
        "ProductVecInvTransform" => (x -> VB.ProductVecInvTransform([Shift(x)], [1:2], (2,))),
        "ReshapeWrapper" => (x -> VB.ReshapeWrapper((1, 2), (2,), Shift(x))),
        "InvReshapeWrapper" => (x -> inverse(VB.ReshapeWrapper((1, 2), (2,), Shift(x)))),
    )
    for (name, f) in factories
        record("$name separate arrays", f([1.0, 2.0]), f([1.0, 2.0]))
        record("$name numeric types", f([1, 2]), f(Float32[1, 2]))
        record("$name signed zero", f([0.0, 2.0]), f([-0.0, 2.0]); expected=false)
        record("$name NaN", f([NaN, 2.0]), f(Float32[NaN, 2.0]))
    end

    for (label, dist) in (
        "mixed" => d,
        "reshaped mixed" => reshape(d, (1, 2)),
        "nested product" => product_distribution([d, deepcopy(d)]),
        "homogeneous fill" => product_distribution(Bijectors.FillArrays.Fill(Beta(), 2)),
    ), f in (VB.to_vec, VB.from_vec, VB.to_linked_vec, VB.from_linked_vec)
        record("$label $f", f(dist), f(deepcopy(dist)))
    end

    s = Stacked([Shift([1.0, 2.0])], [1:2])
    same = typeof(s)(deepcopy(s.bs), copy(s.ranges_in), copy(s.ranges_out), 99, 99)
    different_in = typeof(s)(s.bs, [2:3], s.ranges_out, s.length_in, s.length_out)
    different_out = typeof(s)(s.bs, s.ranges_in, [2:3], s.length_in, s.length_out)
    record("Stacked derived lengths", s, same)
    record("Stacked input ranges", s, different_in; expected=false)
    record("Stacked output ranges", s, different_out; expected=false)
    a = Stacked([identity], [1:0])
    b = Stacked([identity, identity], [1:0, 1:0])
    println("Stacked broadcast cardinality => ", (; eq=a == b, isequal=isequal(a, b),
        set_length=length(Set(Any[a, b]))))
    c = Stacked(view([identity], :), [1:1])
    println("Stacked view reflexivity => ", (; eq=c == c, isequal=isequal(c, c)))

    for f in (Bijectors.CDFBijector, Bijectors.QuantileBijector)
        a, b = f(Normal(0f0, 1f0)), f(Normal(0.0, 1.0))
        println(f, " residual legacy scope => ", (; eq=isequal(a, b), hashes=hash(a) == hash(b)))
    end
end

function run_regressions(source)
    try
        include_string(Main, source, "PR equality tests")
        return nothing
    catch err
        while err isa LoadError
            err = err.error
        end
        err isa Test.TestSetException || rethrow()
        return err
    end
end

const TESTS = read(joinpath(@__DIR__, "..", "head-tests", "test", "bijectors", "equality.jl"), String)

const MUTANT = get(ENV, "REVIEW_MUTANT", "")
if MUTANT == "all-stacked-fields"
    @eval Base.isequal(a::Stacked, b::Stacked) = isequal(Bijectors._transform_fields(a), Bijectors._transform_fields(b))
    @eval Base.hash(a::Stacked, h::UInt) = hash(Bijectors._transform_fields(a), hash(:Stacked, h))
elseif MUTANT == "omit-output-ranges"
    @eval Base.isequal(a::Stacked, b::Stacked) = isequal((a.bs, a.ranges_in), (b.bs, b.ranges_in))
    @eval Base.hash(a::Stacked, h::UInt) = hash((:Stacked, a.bs, a.ranges_in), h)
elseif MUTANT == "logit-equals"
    @eval Base.isequal(a::Logit, b::Logit) = a == b
end

if !isempty(MUTANT)
    @test run_regressions(TESTS) === nothing
    summary = if MUTANT == "logit-equals"
        isequal(Logit(NaN, 1.0), Logit(NaN, 1.0))
    else
        s = Stacked([identity])
        other = MUTANT == "all-stacked-fields" ?
            typeof(s)(s.bs, s.ranges_in, s.ranges_out, 2, 2) :
            typeof(s)(s.bs, s.ranges_in, [2:2], s.length_in, s.length_out)
        isequal(s, other) == (MUTANT == "all-stacked-fields")
    end
    @test !summary
    println("Original assertions missed mutant: ", MUTANT)
elseif HEAD
    @test run_regressions(TESTS) === nothing
else
    for block in split(TESTS, r"(?=@testset \"Hashing\")")
        failure = run_regressions(block)
        @test failure isa Test.TestSetException
        @test failure.error == 0
        println("Base failures in ", first(split(block, '\n')), " => ", failure.fail)
    end
end
