using JuliaBUGS, ADTypes, ReverseDiff, AdvancedHMC, AbstractMCMC, StableRNGs, LogDensityProblems
using Pkg

println("ReverseDiff ", pkgversion(ReverseDiff), ", AbstractMCMC ", pkgversion(AbstractMCMC))

model_def = @bugs begin
    mu ~ dnorm(0, 0.0001)
    tau ~ dgamma(0.01, 0.01)
    sigma = 1 / sqrt(tau)
    for i in 1:N
        x[i] ~ dnorm(mu, tau)
    end
end
x = 5.0 .+ 2.0 .* randn(StableRNG(42), 10)
ad_model = compile(model_def, (N=10, x=x), (mu=0.0, tau=1.0); adtype=AutoReverseDiff(; compile=false))

# Every InstructionTape reachable from `root`, with its field path, whether it is recording,
# and how many TapeNodes its linked list holds.
function tapes(root)
    found = Tuple{String,Bool,Int}[]
    seen = IdDict{Any,Nothing}()
    function walk(x, path)
        (isbits(x) || x isa Module || x isa DataType || x isa String || x isa Symbol) && return
        haskey(seen, x) && return
        seen[x] = nothing
        if x isa ReverseDiff.InstructionTape
            n, node = 0, (@atomic x.last)
            while node isa ReverseDiff.TapeNode
                n += 1
                node = node.prev
            end
            push!(found, (path, ReverseDiff.isrecording(x), n))
            return
        end
        if x isa AbstractArray
            for (i, e) in enumerate(x)
                isassigned(x, i) && walk(e, "$path[$i]")
                i > 50 && break
            end
        elseif x isa AbstractDict
            for (k, v) in x
                walk(v, "$path[$(repr(k))]")
            end
        else
            for f in fieldnames(typeof(x))
                isdefined(x, f) && walk(getfield(x, f), "$path.$f")
            end
        end
    end
    walk(root, "model")
    return found
end

function try_deepcopy(label)
    println("\n== $label")
    for (path, rec, n) in tapes(ad_model)
        println("  tape at $path  recording=$rec  linked nodes=$n")
    end
    try
        deepcopy(ad_model)
        println("  deepcopy(model): ok")
    catch e
        println("  deepcopy(model): ", typeof(e))
    end
end

try_deepcopy("after compile")
LogDensityProblems.logdensity_and_gradient(ad_model, [0.5, 0.1])
try_deepcopy("after one gradient")
sample(StableRNG(123), ad_model, NUTS(0.8), 50; progress=false, n_adapts=20, discard_initial=20)
try_deepcopy("after a single-chain sample")
try
    sample(StableRNG(123), ad_model, NUTS(0.8), MCMCThreads(), 50, 2; progress=false, n_adapts=20, discard_initial=20)
    println("\nMCMCThreads sample: ok")
catch e
    println("\nMCMCThreads sample: ", typeof(e))
end
