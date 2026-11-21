using Bijectors, Test
using Bijectors: Shift, Scale

const VB = Bijectors.VectorBijectors
const mode = only(ARGS)

function check_file(path)
    source = match(r"@testset \"Hashing\" begin.*\z"s, read(path, String)).match
    try
        include_string(@__MODULE__, source, path)
        return nothing
    catch err
        while err isa LoadError
            err = err.error
        end
        return err
    end
end

if mode == "mutant"
    for T in (Scale, Bijectors.Inverse, VB.ProductVecTransform,
              VB.ProductVecInvTransform, VB.ReshapeWrapper, VB.InvReshapeWrapper)
        @eval Base.hash(b::$T, h::UInt) =
            hash((typeof(b), ntuple(i -> getfield(b, i), fieldcount(typeof(b)))), h)
    end
    @test check_file("original-equality.jl") === nothing
    result = check_file("target/test/bijectors/equality.jl")
    @test result isa Test.TestSetException
    @test result.fail > 0 && result.error == 0
    println("The old tests accept concrete-type hashes. The revised tests reject them.")
    exit()
end

module OriginalExample
    using Bijectors
    const result = try
        b = Shift([1.0, 2.0])
        length(Set([b, deepcopy(b)]))
    catch err
        err
    end
end
@test OriginalExample.result isa UndefVarError

module FixedExample
    using Bijectors: Shift
    const b = Shift([1.0, 2.0])
    const result = length(Set([b, deepcopy(b)]))
end
@test FixedExample.result == (mode == "base" ? 2 : 1)
println("Corrected PR example: ", FixedExample.result)

result = check_file("target/test/bijectors/equality.jl")
if mode == "base"
    @test result isa Test.TestSetException
    @test result.fail > 0 && result.error == 0
    println("Revised regression failures on base: ", result.fail)
else
    @test result === nothing
    for f in (Shift, Scale, x -> Bijectors.Inverse(Shift(x)))
        for values in ([NaN, -0.0], [missing, 1], ComplexF64[1 + 2im])
            a, b = f(values), f(deepcopy(values))
            @test isequal(a, b)
            @test hash(a, hash(:seed)) == hash(b, hash(:seed))
            @test length(Set([a, b])) == 1
        end
    end
    for d in (product_distribution([Normal(), Beta(2, 2)]),
              product_distribution(fill(Beta(2, 2), 2)),
              reshape(product_distribution([Normal(), Beta(2, 2)]), (1, 2)))
        for f in (VB.to_vec, VB.from_vec, VB.to_linked_vec, VB.from_linked_vec)
            a, b = f(d), f(d)
            @test isequal(a, b)
            @test hash(a) == hash(b)
            @test get(Dict{Any,Int}(a => 1), b, 0) == 1
        end
    end
end
