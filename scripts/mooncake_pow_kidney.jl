# Mooncake 0.5.70 against 0.5.71 (chalk-lab/Mooncake.jl#1477) on the `^` benchmark from #1333 and on
# the gradient of JuliaBUGS's Kidney example, plus 0.5.71 with the direct rule proposed in #1333.
# Each measurement is a fresh process, alternating between versions on one machine.
if get(ARGS, 1, "") == "bench"
    using Mooncake, Chairmarks, Random, JuliaBUGS, LogDensityProblems, ADTypes

    if ARGS[3] == "direct"
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
        "RESULT ", ARGS[2], " ", ARGS[3], " pow_sum_us=", round(t_pow * 1e6; digits=3),
        " kidney_us=", round(t_kidney * 1e6; digits=3), " kidney_logp=", value,
        " kidney_grad_l1=", gradient_l1, " mooncake=", pkgversion(Mooncake),
    )
else
    julia = Base.julia_cmd()
    jb = abspath("jb/JuliaBUGS")
    envs = Dict{String,String}()
    for version in ("0.5.70", "0.5.71")
        env = mktempdir()
        code = """
            using Pkg
            Pkg.develop(path=$(repr(jb)))
            Pkg.add(["Chairmarks", "LogDensityProblems", "ADTypes"])
            Pkg.add(name="Mooncake", version=$(repr(version)))
            Pkg.pin("Mooncake")
            Pkg.precompile()
            """
        run(`$julia --project=$env -e $code`)
        envs[version] = env
    end

    println("Julia ", VERSION)
    results = Dict{String,Vector{Tuple{Float64,Float64}}}()
    for _ in 1:3, (version, rule) in (("0.5.70", "nfwd"), ("0.5.71", "nfwd"), ("0.5.71", "direct"))
        out = read(`$julia --project=$(envs[version]) $(abspath(@__FILE__)) bench $version $rule`, String)
        line = only(filter(startswith("RESULT"), split(out, '\n')))
        println(line)
        fields = Dict(k => v for (k, v) in split.(split(line)[4:end], '='))
        push!(
            get!(results, "$version $rule", Tuple{Float64,Float64}[]),
            (parse(Float64, fields["pow_sum_us"]), parse(Float64, fields["kidney_us"])),
        )
    end

    println("\nminimum over runs (us):")
    for key in ("0.5.70 nfwd", "0.5.71 nfwd", "0.5.71 direct")
        println(
            "  ", rpad(key, 14), " pow_sum ", minimum(first.(results[key])),
            "  kidney ", minimum(last.(results[key])),
        )
    end
end
