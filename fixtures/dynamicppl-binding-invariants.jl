using DynamicPPL, Distributions, Random, Test

revision = only(ARGS)
@assert realpath(dirname(dirname(pathof(DynamicPPL)))) == realpath(revision)
@show revision VERSION

function probe(f, name)
    try
        println(name, " = ", repr(f()))
    catch err
        println(name, " = ", sprint(showerror, err))
    end
    return nothing
end

@model observed_child(x) = (x ~ Normal(); x)
@model wrap_child(child) = (a ~ to_submodel(child); a)
child = observed_child(2.0)
explicit = condition(child; x=2.0)
probe("argument and explicit observation compare equal") do
    child == explicit, isequal(child, explicit), hash(child) == hash(explicit)
end
probe(() -> wrap_child(child)(Xoshiro(4)), "child with an argument observation")
probe(() -> wrap_child(explicit)(Xoshiro(4)), "child with an explicit observation")

@model function shadowed(x)
    let x = missing
        x ~ Normal()
        return x
    end
end
@model function renamed(data)
    let x = missing
        x ~ Normal()
        return x
    end
end
@model function loop_shadowed(x)
    for x in (missing,)
        x ~ Normal()
        return x
    end
end
probe(() -> shadowed(2.0)(Xoshiro(4)), "shadowed argument")
probe(() -> renamed(2.0)(Xoshiro(4)), "renamed argument")
probe(() -> loop_shadowed(2.0)(Xoshiro(4)), "shadowed by a loop")

@model scalar_child() = (x ~ Normal(); x)
@model function indexed_child(child)
    p = zeros(2, 2)
    p[3] ~ to_submodel(child)
    return p[3]
end
for (bind, remove, role) in ((condition, decondition, "condition"), (fix, unfix, "fix"))
    model = wrap_child(indexed_child(bind(scalar_child(); x=2.0)))
    probe("$role removal via linear and Cartesian addresses") do
        linear = remove(model, @varname(a.p[3].x))
        cart = remove(model, @varname(a.p[1, 2].x))
        Tuple(keys(rand(Xoshiro(4), linear))), Tuple(keys(rand(Xoshiro(4), cart)))
    end
end

@model function two_positions(x)
    for i in eachindex(x)
        x[i] ~ Normal()
    end
    return x
end
if revision == "head"
    @testset "Independent roles and fresh removal" begin
        model = two_positions([0.0, 0.0])
        latent = decondition(model)
        a = condition(fix(latent, @varname(x[2]) => 2.0), @varname(x[1]) => 1.0)
        b = fix(condition(latent, @varname(x[1]) => 1.0), @varname(x[2]) => 2.0)
        @test a(Xoshiro(4)) == b(Xoshiro(4)) == [1.0, 2.0]
        @test logjoint(a, (;)) == logjoint(b, (;)) == logpdf(Normal(), 1.0)
        @test keys(rand(Xoshiro(4), decondition(a, @varname(x[1])))) == [@varname(x[1])]
        @test keys(rand(Xoshiro(4), unfix(a, @varname(x[2])))) == [@varname(x[2])]
        @test latent(Xoshiro(4)) == decondition(unfix(a))(Xoshiro(4))
        @test model(Xoshiro(4)) == [0.0, 0.0]
    end
end
