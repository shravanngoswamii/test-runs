# Precompile JuliaBUGS in one depot, move the depot, and read examples from the moved copy.
julia = Base.julia_cmd()
jb = abspath("jb")
run(`git -C $jb fetch --quiet --unshallow`)
rev = readchomp(`git -C $jb rev-parse HEAD`)
a = joinpath(mktempdir(), "depot")
b = joinpath(mktempdir(), "depot")

function in_depot(depot, code)
    cmd = `$julia --project=$(joinpath(depot, "env")) -e $code`
    return run(addenv(cmd, "JULIA_DEPOT_PATH" => depot * ":"))
end

in_depot(a, """
    using Pkg
    Pkg.add(url="$jb", subdir="JuliaBUGS", rev="$rev")
    Pkg.precompile()
    """)
mv(a, b)
in_depot(b, """
    t = @elapsed using JuliaBUGS
    println("using JuliaBUGS took ", round(t; digits=1), " s from ", pathof(JuliaBUGS))
    for key in (:rats, :jama, :methadone)
        result = try
            JuliaBUGS.BUGSExamples.load(key).name
        catch err
            first(sprint(showerror, err), 200)
        end
        println(key, ": ", result)
    end
    """)
