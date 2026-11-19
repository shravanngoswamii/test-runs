using Bijectors
using Distributions
using DynamicPPL
using DynamicPPL.VarNamedTuples: PartialArray
using OffsetArrays
using Random: Xoshiro
using TOML

rows = Dict{String,Any}[]
function inspect_pair(label, a, b)
    row = Dict{String,Any}("label" => label, "type" => string(typeof(a)))
    try
        row["equal"] = isequal(a, b)
        row["hashes_agree"] = all(h -> hash(a, h) == hash(b, h), UInt[0, 1, typemax(UInt)])
        row["set_length"] = length(Set(Any[a, b]))
        row["dict_lookup"] = get(Dict{Any,Int}(a => 1), b, 0)
        row["hash_method"] = string(which(hash, (typeof(a), UInt)))
    catch err
        row["error"] = sprint(showerror, err)
    end
    push!(rows, row)
    println(row)
end

for transform in (
    Bijectors.Shift([1.0, 2.0]),
    Bijectors.Scale([2.0, 3.0]),
    Bijectors.VectorBijectors.from_linked_vec(product_distribution([Normal(), Beta(2, 2)])),
    Bijectors.VectorBijectors.from_linked_vec(Dirichlet([1.0, 2.0, 3.0])),
)
    other = deepcopy(transform)
    label = string(typeof(transform))
    inspect_pair("child $label", transform, other)
    inspect_pair("FixedTransform $label", FixedTransform(transform), FixedTransform(other))
    inspect_pair("TransformedValue $label", TransformedValue([0.1, 0.2], FixedTransform(transform)), TransformedValue([0.1, 0.2], FixedTransform(other)))
    inspect_pair("WithTransforms $label", WithTransforms(VarNamedTuple(; travel=FixedTransform(transform)), UnlinkAll()), WithTransforms(VarNamedTuple(; travel=FixedTransform(other)), UnlinkAll()))
end

@model function product_model()
    travel ~ product_distribution([Normal(), Beta(2, 2)])
    return travel
end
model = product_model()
first = get_fixed_transforms(model, LinkAll())
second = get_fixed_transforms(model, LinkAll())
inspect_pair("model fixed transforms", first, second)
strategy1 = WithTransforms(first, UnlinkAll())
strategy2 = WithTransforms(second, UnlinkAll())
inspect_pair("model fixed strategies", strategy1, strategy2)
params = InitFromParams((; travel=[0.1, 0.5]), nothing)
a = VarInfo(Xoshiro(1), model, params, strategy1)
b = VarInfo(Xoshiro(1), model, params, strategy2)
inspect_pair("model fixed VarInfo", a, b)

for start in (0, -2)
    data = OffsetArray([1, 2, 3], start:(start + 2))
    mask = OffsetArray([true, false, true], start:(start + 2))
    a = PartialArray(data, mask)
    b = PartialArray(Float64.(data), copy(mask))
    inspect_pair("offset PartialArray $start", a, b)
end
for numeric in (0, 0.0, -0.0, 0f0)
    inspect_pair("NoLogProb against $numeric", LogPriorAccumulator(DynamicPPL.NoLogProb()), LogPriorAccumulator(numeric))
end

open("results.toml", "w") do io
    TOML.print(io, Dict(
        "julia" => string(VERSION), "bijectors" => string(Base.pkgversion(Bijectors)),
        "head" => get(ENV, "REVIEW_HEAD", "false") == "true", "observations" => rows,
    ))
end
