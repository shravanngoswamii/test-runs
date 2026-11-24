using Turing, Turing.Variational, ReverseDiff, Optimisers, AdvancedVI, ForwardDiff, Test
using Random: Xoshiro

@model function gdemo_d()
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
    return s, m
end

println("Optimisers ", pkgversion(Optimisers), ", ReverseDiff ", pkgversion(ReverseDiff),
        ", ForwardDiff ", pkgversion(ForwardDiff), ", AdvancedVI ", pkgversion(AdvancedVI))
@testset "VI on $(ARGS[1]) with Optimisers $(pkgversion(Optimisers))" begin
    for family in (q_meanfield_gaussian, q_fullrank_gaussian)
        outcome = try
            vi(Xoshiro(42), gdemo_d(), family, 2; adtype=AutoReverseDiff(), show_progress=false)
        catch err
            println(family, ": ", sprint(showerror, err))
            err
        end
        kind = outcome isa MethodError ? :method_error : outcome isa Turing.Variational.VIResult ? :success : :other
        println(family, ": ", kind)
        if ARGS[2] == "fails" && family === q_fullrank_gaussian
            @test kind === :method_error
            @test occursin("Float64(::ForwardDiff.Dual", sprint(showerror, outcome))
        else
            @test kind === :success
        end
    end
end
