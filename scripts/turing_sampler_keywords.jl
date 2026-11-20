using Turing, Random, Test, TOML, REPL
import DynamicPPL

@model scalar_normal() = x ~ Normal()

const results = Dict{String,Any}()

function run_case(label, sampler, n; first, adaptations, kwargs...)
    indices = Int[]
    chain = sample(
        Xoshiro(468), scalar_normal(), sampler, n;
        progress=false, verbose=false, save_state=true,
        callback=(_, _, _, _, state, _; kwargs...) -> push!(indices, state.i),
        kwargs...,
    )
    state = only(loadstate(chain))
    count = hasproperty(state.adaptor, :state) ? state.adaptor.state.i : 0
    @test size(chain, 1) == n
    @test indices[1] == first
    @test count == adaptations
    results[label] = Dict("values" => vec(chain[@varname(x)]), "indices" => indices,
                          "adaptations" => count)
    println(label, ": first transition = ", indices[1], ", adaptations = ", count,
            ", retained = ", length(indices))
    return chain
end

@testset "Adaptive sampling documentation claims" begin
    for (name, make_sampler, default_sampler) in (
        ("NUTS", n -> NUTS(n, 0.65; init_ϵ=0.1), () -> NUTS(; init_ϵ=0.1)),
        ("HMCDA", n -> HMCDA(n, 0.65, 0.3; init_ϵ=0.1),
         () -> HMCDA(0.65, 0.3; init_ϵ=0.1)),
    )
        run_case("$name/default", default_sampler(), 20; first=10, adaptations=10)
        run_case("$name/cap", default_sampler(), 2002; first=1000, adaptations=1000)
        run_case("$name/constructor", make_sampler(8), 20; first=8, adaptations=8)
        run_case("$name/override", make_sampler(8), 20; nadapts=4, first=4, adaptations=4)
        run_case("$name/auto", make_sampler(8), 20; nadapts=-1, first=10, adaptations=10)
        run_case("$name/zero", default_sampler(), 20; nadapts=0, first=0, adaptations=0)
        run_case("$name/constructor-zero", make_sampler(0), 20;
                 nadapts=4, first=4, adaptations=0)
        full = run_case("$name/keep", make_sampler(8), 40;
                        discard_adapt=false, first=0, adaptations=8)
        run_case("$name/short", make_sampler(8), 4;
                 discard_adapt=false, first=0, adaptations=3)
        for flag in (false, true)
            c = run_case("$name/explicit-$flag", make_sampler(8), 20;
                         discard_adapt=flag, discard_initial=2, first=2, adaptations=8)
            @test vec(c[@varname(x)]) == vec(full[@varname(x)])[3:22]
        end
        c = run_case("$name/thinning", make_sampler(8), 10;
                     thinning=3, first=8, adaptations=8)
        @test vec(c[@varname(x)]) == vec(full[@varname(x)])[9:3:36]
        run_case("$name/warmup", make_sampler(8), 20;
                 num_warmup=12, nadapts=4, first=4, adaptations=4)
        @test results["$name/warmup"] == results["$name/override"]
        @test results["$name/constructor"]["values"] == vec(full[@varname(x)])[9:28]

        state = only(loadstate(full))
        resumed = [sample(Xoshiro(900), scalar_normal(), make_sampler(8), 5;
                          initial_state=deepcopy(state), nadapts=k, discard_initial=k,
                          discard_adapt=flag, progress=false, verbose=false, save_state=true)
                   for (k, flag) in ((0, false), (10, true))]
        @test resumed[1][@varname(x)] == resumed[2][@varname(x)]
        @test only(loadstate(resumed[1])).i == state.i + 5
        @test only(loadstate(resumed[1])).adaptor.state.i == 8
        results["$name/resume"] = vec(resumed[1][@varname(x)])

        for mode in (MCMCSerial(), MCMCThreads())
            # Seeded equality is checked within each execution mode.
            chains = [sample(Xoshiro(112), scalar_normal(), make_sampler(8), mode, 10, 2;
                             nadapts=4, progress=false, verbose=false, save_state=true)
                      for _ in 1:2]
            @test chains[1][@varname(x)] == chains[2][@varname(x)]
            @test all(s -> s.i == 13 && s.adaptor.state.i == 4, loadstate(chains[1]))
            results["$name/$(nameof(typeof(mode)))"] = vec(chains[1][@varname(x)])
        end
    end

    # The no-RNG entry point uses Random.default_rng().
    Random.seed!(137)
    a = sample(scalar_normal(), NUTS(), 4; progress=false, verbose=false)
    Random.seed!(137)
    b = sample(Random.default_rng(), scalar_normal(), NUTS(), 4;
               progress=false, verbose=false)
    @test a[@varname(x)] == b[@varname(x)]
    results["default-rng"] = vec(a[@varname(x)])
end

helptext = sprint(show, MIME"text/plain"(), @doc sample)
write("$(only(ARGS))-help.txt", helptext)
@test occursin("For a new chain, the following keywords", helptext) == (only(ARGS) == "head")
open("$(only(ARGS))-results.toml", "w") do io
    TOML.print(io, results; sorted=true)
end
if only(ARGS) == "head"
    @test results == TOML.parsefile("base-results.toml")
    println("Every recorded value and transition index agrees between base and head.")
end
