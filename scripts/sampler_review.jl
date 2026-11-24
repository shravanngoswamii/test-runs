using Turing, Random, Test, TOML, REPL

@model scalar_normal() = x ~ Normal()
mode = only(ARGS)
results = Dict{String,Any}()

@testset "Zero constructor count overrides nadapts" begin
    for sampler in (NUTS(0, 0.65), HMCDA(0, 0.65, 0.3))
        stepsizes = Float64[]
        chain = sample(
            Xoshiro(468), scalar_normal(), sampler, 10;
            nadapts=4, discard_adapt=false, progress=false, verbose=false,
            callback=(_, _, _, _, state, _; kwargs...) -> begin
                hasproperty(state.stat, :step_size) && push!(stepsizes, state.stat.step_size)
            end,
        )
        @test length(stepsizes) == 9
        @test all(==(first(stepsizes)), stepsizes)
        results[string(nameof(typeof(sampler)))] = vec(chain[@varname(x)])
    end
end

@testset "Effective discard defaults" begin
    for (name, sampler) in (("NUTS", NUTS(4, 0.65; init_ϵ=0.1)),
                            ("HMCDA", HMCDA(4, 0.65, 0.3; init_ϵ=0.1)))
        kept = sample(Xoshiro(5), scalar_normal(), sampler, 14;
                      discard_adapt=false, progress=false, verbose=false)
        default = sample(Xoshiro(5), scalar_normal(), sampler, 10;
                         progress=false, verbose=false)
        explicit = sample(Xoshiro(5), scalar_normal(), sampler, 10;
                          discard_initial=2, progress=false, verbose=false)
        @test size(default, 1) == 10
        @test vec(default[@varname(x)]) == vec(kept[@varname(x)])[5:14]
        @test vec(explicit[@varname(x)]) == vec(kept[@varname(x)])[3:12]
        results["$name/default"] = vec(default[@varname(x)])
    end
end

helptext = sprint(show, MIME"text/plain"(), @doc sample)
write("$mode-help.txt", helptext)
if mode == "head"
    plain = join(split(helptext), ' ')
    @test occursin("never adapt, regardless of", plain)
    @test occursin("discard_initial=discard_adapt ? nadapts : 0", plain)
    @test results == TOML.parsefile("previous-sampler-values.toml")
end
open("$mode-sampler-values.toml", "w") do io
    TOML.print(io, results; sorted=true)
end
println("Sampler review passed for ", mode)
