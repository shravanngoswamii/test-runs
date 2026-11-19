# One measurement for mooncake_pow_kidney.jl: `julia --project=<env> mooncake_pow_kidney_bench.jl <label> <nfwd|direct>`
using Mooncake, Chairmarks, Random, JuliaBUGS, LogDensityProblems, ADTypes

if ARGS[2] == "direct"
    @eval function Mooncake.rrule!!(
        ::Mooncake.CoDual{typeof(^)}, x::Mooncake.CoDual{Float64}, p::Mooncake.CoDual{Float64}
    )
        _x, _p = Mooncake.primal(x), Mooncake.primal(p)
        y = _x^_p
        gx = Mooncake.Nfwd._nfwd_pow_grad_x(_x, _p, y)
        gp = ifelse(!iszero(_x), y * log(abs(_x)), ifelse(_p > 0, zero(y), NaN))
        pow_pullback(dy::Float64) = (Mooncake.NoRData(), dy * gx, dy * gp)
        return Mooncake.zero_fcodual(y), pow_pullback
    end
end

pow_sum(z) = sum(z[i]^z[end] for i in 1:(length(z) - 1))

function time_pow_sum()
    z = vcat(0.5 .+ rand(Xoshiro(1), 100), 1.3)
    cache = Mooncake.prepare_gradient_cache(pow_sum, z)
    Mooncake.value_and_gradient!!(cache, pow_sum, z)
    return minimum(@be Mooncake.value_and_gradient!!(cache, pow_sum, z) seconds = 2).time
end

function time_kidney()
    ex = JuliaBUGS.BUGSExamples.VOLUME_1.kidney
    model = compile(ex.model_def, ex.data, ex.inits)
    model = JuliaBUGS.set_evaluation_mode(model, JuliaBUGS.UseGeneratedLogDensityFunction())
    grad_model = JuliaBUGS.BUGSModelWithGradient(model, AutoMooncake(; config=nothing))
    θ = JuliaBUGS.getparams(model)
    value, gradient = LogDensityProblems.logdensity_and_gradient(grad_model, θ)
    t = minimum(@be LogDensityProblems.logdensity_and_gradient(grad_model, θ) seconds = 2).time
    return t, value, sum(abs, gradient)
end

t_pow = time_pow_sum()
t_kidney, value, gradient_l1 = time_kidney()
println(
    "RESULT ", ARGS[1], " ", ARGS[2], " pow_sum_us=", round(t_pow * 1e6; digits=3),
    " kidney_us=", round(t_kidney * 1e6; digits=3), " kidney_logp=", value,
    " kidney_grad_l1=", gradient_l1, " mooncake=", pkgversion(Mooncake),
)
