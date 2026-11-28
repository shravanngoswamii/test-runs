using AbstractMCMC: SamplingOutput
using DynamicPPL, Random, StableRNGs, Turing
using Turing.Inference: Particle, advance!, particle_rng
using Test: @testset

function trace_invokes(model)
    inf = Turing.Inference
    lt = inf.Libtask
    model = DynamicPPL.setleafcontext(model, inf.SMCContext())
    args, kwargs = DynamicPPL.make_evaluate_args_and_kwargs(model, inf.particle_varinfo())
    for f in (inf.evaluate_particle!, model.f)
        println("Input IR for ", typeof(f))
        bb = lt.generate_ir(:input_bb, f, args...; kwargs...)
        for block in bb.blocks, inst in block.insts
            stmt = inst.stmt
            if Meta.isexpr(stmt, :invoke)
                mi = lt.get_mi(stmt.args[1])
                name = mi.def isa Method ? mi.def.name : Symbol("unknown")
                mod = mi.def isa Method ? mi.def.module : :unknown
                println("  invoke ", mod, ".", name, " tape=", lt.stmt_might_produce(stmt, lt.CC.widenconst(inst.type)))
            elseif Meta.isexpr(stmt, :call)
                println("  call ", stmt.args[1], " tape=", lt.stmt_might_produce(stmt, lt.CC.widenconst(inst.type)))
            end
        end
    end
end

@model function dynamic_bernoulli_normal(y_obs=2.0)
    b ~ Bernoulli(0.3)
    θ = zeros(2)
    if b == 0
        θ[1] ~ Normal()
        y_obs ~ Normal(θ[1], 0.5)
    else
        θ[1] ~ Normal()
        θ[2] ~ Normal()
        y_obs ~ Normal(θ[1] + θ[2], 0.5)
    end
end
function diagnose(model, seed)
    particle = Particle(model, particle_rng(Xoshiro(seed)))
    scores = Float64[]
    while (score = advance!(particle)) !== nothing
        push!(scores, score)
    end
    vi = particle.varinfo
    vals = DynamicPPL.get_parameter_values(vi)
    @show seed scores keys(vals) DynamicPPL.getloglikelihood(vi) loglikelihood(model, vals)
    flush(stdout)
end

model = dynamic_bernoulli_normal(2.0)
@show VERSION pathof(DynamicPPL) pathof(Turing) isdefined(DynamicPPL, :BindingTable)
only(ARGS) == "trace-invokes" && trace_invokes(model)
for seed in 1:5
    diagnose(model, seed)
end

function compare_draws(model)
    for sampler in (
        Gibbs((@varname(b), @varname(θ)) => PG(50)),
        Gibbs((@varname(b), @varname(θ)) => PG(20), @varname(θ) => MH()),
    )
        kwargs = (; discard_initial=1_000, progress=false, verbose=false)
        for chain_type in (SamplingOutput, VNChain)
            out = sample(StableRNG(42), model, sampler, 2_000; kwargs..., chain_type)
            bs = chain_type === SamplingOutput ?
                [d.params[@varname(b)] for d in out.samples] : out[@varname(b)]
            @show sampler chain_type count(==(1), bs) / length(bs)
            flush(stdout)
        end
    end
end

@testset "Gibbs in test scope" begin
    @model function dynamic_bernoulli_normal_local(y_obs=2.0)
        b ~ Bernoulli(0.3)
        θ = zeros(2)
        if b == 0
            θ[1] ~ Normal(0.0, 1.0)
            y_obs ~ Normal(θ[1], 0.5)
        else
            θ[1] ~ Normal(0.0, 1.0)
            θ[2] ~ Normal(0.0, 1.0)
            y_obs ~ Normal(θ[1] + θ[2], 0.5)
        end
    end
    model = dynamic_bernoulli_normal_local(2.0)
    only(ARGS) == "trace-invokes" && trace_invokes(model)
    for seed in 1:5
        diagnose(model, seed)
    end
    compare_draws(model)
end
