# Checks for TuringLang/JuliaBUGS.jl#574: latent and observed `dmulti` under auto-marginalization,
# and Gibbs validation of a gradient sampler on a latent `dmulti`.
using JuliaBUGS, LogDensityProblems, Distributions, LogExpFunctions
using AdvancedHMC, ADTypes, ForwardDiff
using JuliaBUGS: @varname, Gibbs
using OrderedCollections: OrderedDict

println("JuliaBUGS ", pkgversion(JuliaBUGS), " at ", pkgdir(JuliaBUGS))
function automarg(model)
    model = JuliaBUGS.settrans(model, true)
    return JuliaBUGS.set_evaluation_mode(model, JuliaBUGS.UseAutoMarginalization())
end

println("\n1. the repro from #561")
dmulti_def = @bugs begin
    for i in 1:3
        y[i, 1:3] ~ dmulti(p[1:3], 1)
    end
end
model = automarg(compile(dmulti_def, (; p=[0.2, 0.3, 0.5])))
println("   mode ", nameof(typeof(model.evaluation_mode)), ", dimension ", LogDensityProblems.dimension(model))

println("\n2. observed dmulti whose probabilities depend on a marginalized dcat")
mixed_def = @bugs begin
    z ~ dcat(w[1:2])
    for j in 1:3
        pz[j] = (2 - z) * q1[j] + (z - 1) * q2[j]
    end
    for i in 1:2
        y[i, 1:3] ~ dmulti(pz[1:3], n[i])
    end
    mu ~ dnorm(0, 1)
    x ~ dnorm(mu, 1)
end
w, q1, q2, n = [0.4, 0.6], [0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [4, 3]
y, x = [1 1 2; 2 1 0], 0.5
model = automarg(compile(mixed_def, (; w, q1, q2, n, y, x)))
lik(q) = sum(logpdf(Multinomial(n[i], q), y[i, :]) for i in 1:2)
for mu in (-0.7, 0.3)
    expected = logsumexp([log(w[1]) + lik(q1), log(w[2]) + lik(q2)]) +
        logpdf(Normal(0, 1), mu) + logpdf(Normal(mu, 1), x)
    got = LogDensityProblems.logdensity(model, [mu])
    println(
        "   mode ", nameof(typeof(model.evaluation_mode)), ", dimension ",
        LogDensityProblems.dimension(model), ", mu = $mu: logdensity $got, by hand $expected, equal ",
        isapprox(got, expected; rtol=1e-12),
    )
end

println("\n3. Gibbs with NUTS on a latent discrete variable")
gibbs_defs = (
    "dmulti" => (@bugs(begin
        mu ~ dnorm(0, 1)
        counts[1:3] ~ dmulti(p[1:3], 1)
        x ~ dnorm(mu + counts[1], 1)
    end), (; p=[0.2, 0.3, 0.5], x=0.5), @varname(counts)),
    "dpois" => (@bugs(begin
        mu ~ dnorm(0, 1)
        k ~ dpois(2)
        x ~ dnorm(mu + k, 1)
    end), (; x=0.5), @varname(k)),
)
for (name, (def, data, latent)) in gibbs_defs
    model = compile(def, data)
    sampler_map = OrderedDict(
        latent => (NUTS(0.8), AutoForwardDiff()), @varname(mu) => (NUTS(0.8), AutoForwardDiff())
    )
    result = try
        Gibbs(model, sampler_map)
        "accepted"
    catch err
        "rejected: " * first(sprint(showerror, err), 160)
    end
    println("   $name latent: ", result)
end
