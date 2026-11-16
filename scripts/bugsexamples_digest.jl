# Prints a digest of every BUGS example's data, inits and reference results, with element types,
# so loads under different JSON.jl or Julia versions can be compared.
using JuliaBUGS, SHA, Pkg

canon(x::NamedTuple) = "NT(" * join(("$k=" * canon(v) for (k, v) in pairs(x)), ",") * ")"
canon(x::AbstractArray) = "A{$(eltype(x))}$(size(x))[" * join(canon.(vec(x)), ",") * "]"
canon(x::AbstractFloat) = "F:" * repr(Float64(x))
canon(x::Integer) = "I:" * string(x)
canon(::Missing) = "missing"
canon(::Nothing) = "nothing"
canon(x::AbstractString) = repr(String(x))
canon(x) = "?$(typeof(x)):" * repr(x)

E = JuliaBUGS.BUGSExamples
println("JSON ", pkgversion(E.JSON), ", Julia ", VERSION)
all_parts = String[]
for e in E.ENTRIES
    ex = E.load(e.path)
    s = join(canon.((ex.data, ex.inits, ex.inits_alternative, ex.reference_results)), "|")
    push!(all_parts, s)
    println(rpad(string(e.key), 28), bytes2hex(sha256(s))[1:16])
end
println("ALL ", bytes2hex(sha256(join(all_parts, "\n")))[1:16])
