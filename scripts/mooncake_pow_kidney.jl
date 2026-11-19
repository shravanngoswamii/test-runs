# Mooncake 0.5.70 against 0.5.71 (chalk-lab/Mooncake.jl#1477) on the `^` benchmark from #1333 and on
# the gradient of JuliaBUGS's Kidney example, plus 0.5.71 with the direct rule proposed in #1333.
# Each measurement is a fresh process, alternating between versions on one machine.
julia = Base.julia_cmd()
jb = abspath("jb/JuliaBUGS")
bench = joinpath(@__DIR__, "mooncake_pow_kidney_bench.jl")
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
    out = read(`$julia --project=$(envs[version]) $bench $version $rule`, String)
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
