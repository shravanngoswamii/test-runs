using Bijectors, Distributions, DynamicPPL, Random, Test

const HEAD = get(ENV, "REVIEW_HEAD", "false") == "true"
println("Julia ", VERSION, " | Bijectors ", pkgversion(Bijectors), " | Distributions ",
    pkgversion(Distributions), " | DynamicPPL ", pkgversion(DynamicPPL))

@model function product_model(d)
    x ~ d
    return x
end

function record(label, a, b)
    eq = isequal(a, b)
    hashes = all(h -> hash(a, h) == hash(b, h), (UInt(0), UInt(1), typemax(UInt)))
    count = length(Set(Any[a, b]))
    found = get(Dict{Any,Int}(a => 1), b, 0)
    println(label, " => ", (; eq, hashes, count, found))
    @test eq
    if HEAD
        @test hashes
        @test count == 1
        @test found == 1
    end
    return hashes
end

@testset "DynamicPPL fixed vector transforms" begin
    a = FixedTransform(Bijectors.Shift([1.0, 2.0]))
    @test record("FixedTransform Shift", a, deepcopy(a)) == HEAD

    d = product_distribution([Normal(), Beta()])
    for (label, dist) in ("mixed" => d, "reshaped" => reshape(d, (1, 2)),
            "nested" => product_distribution([d, deepcopy(d)]))
        m = product_model(dist)
        a = get_fixed_transforms(m, LinkAll())
        b = get_fixed_transforms(m, LinkAll())
        @test record("$label fixed transform", a[@varname(x)], b[@varname(x)]) == HEAD
        record("$label fixed storage", a, b)
        sa = WithTransforms(a, UnlinkAll())
        sb = WithTransforms(b, UnlinkAll())
        record("$label fixed strategy", sa, sb)
        raw = rand(Xoshiro(1), dist)
        init = InitFromParams((; x=raw), nothing)
        va = VarInfo(Xoshiro(2), m, init, sa)
        vb = VarInfo(Xoshiro(2), m, init, sb)
        record("$label VarInfo", va, vb)
    end
end
