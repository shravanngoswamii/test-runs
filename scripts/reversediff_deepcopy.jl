using JuliaBUGS, ADTypes, DifferentiationInterface, ReverseDiff, AdvancedHMC, AbstractMCMC
using StableRNGs, LogDensityProblems

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
        # what `deepcopy` returns as is
        (
            isbits(x) || x isa Union{
                Module,String,Symbol,Core.MethodInstance,Method,GlobalRef,DataType,Union,
                UnionAll,Task,Regex,
            }
        ) && return
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
        if x isa Union{AbstractArray,Memory}
            for i in eachindex(x)
                isassigned(x, i) && walk(x[i], "$path[$i]")
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

println("threads: ", Threads.nthreads())
try_deepcopy("after compile")
copy1 = deepcopy(ad_model)
input(m) = m.prep.cache.gradient_prep.config.input
println("  a copy shares the tracked input: ", input(copy1) === input(ad_model))
sample(StableRNG(123), ad_model, NUTS(0.8), 200; progress=false, n_adapts=100, discard_initial=100)
try_deepcopy("after a single-chain sample")
for k in 1:3
    try
        sample(StableRNG(123), ad_model, NUTS(0.8), MCMCThreads(), 200, 2; progress=false, n_adapts=100, discard_initial=100)
        println("\nMCMCThreads sample $k: ok")
    catch e
        println("\nMCMCThreads sample $k: ", typeof(e))
    end
    try_deepcopy("after MCMCThreads sample $k")
end
