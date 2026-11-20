using Test

source = read("head-tests/test/debug_utils.jl", String)
first_line = findfirst("    @testset \"non-numeric data\" begin\n", source)
isnothing(first_line) && error("Regression testset was not found")
last_line = findnext("    end", source, last(first_line) + 1)
fixture = match(r"struct NaNObservation end\nBase\.isnan[^\n]+\n", source).match
regression = fixture * source[first(first_line):last(last_line)]

if !isempty(ARGS)
    using DynamicPPL
    if ARGS[1] == "mute"
        @eval DynamicPPL.DebugUtils _has_nans(x::Number) = false
    elseif ARGS[1] == "number_only"
        @eval DynamicPPL.DebugUtils begin
            _has_nans(x::Number) = isnan(x)
            _has_nans(x) = false
        end
    elseif ARGS[1] == "symbol_only"
        @eval DynamicPPL.DebugUtils begin
            _has_nans(x) = isnan(x)
            _has_nans(::Symbol) = false
            _has_nans(::Missing) = false
        end
    end
    include_string(Main, regression, "head regression")
else
    fixed = parse(Bool, ENV["REVIEW_HEAD"])
    @testset "Regression distinguishes the fix" begin
        for mode in (fixed ? ("correct", "mute", "symbol_only", "number_only") : ("base",))
            file = "regression-$(mode).log"
            process = open(file, "w") do io
                run(pipeline(ignorestatus(`$(Base.julia_cmd()) --project=target/test $(@__FILE__) $mode`),
                             stdout=io, stderr=io))
            end
            output = read(file, String)
            print(output)
            if mode == "correct"
                @test success(process)
                @test occursin(r"non-numeric data\s+\|\s+3\s+3", output)
            else
                @test !success(process)
                @test occursin(mode in ("mute", "number_only") ? "Test Failed" : "Error During Test", output)
                if mode in ("base", "symbol_only")
                    @test occursin("isnan", output)
                end
            end
        end
    end
end
