# Checks that editing or adding BUGS example files in a developed JuliaBUGS checkout rebuilds the package cache.
using JuliaBUGS

root = joinpath(pkgdir(JuliaBUGS), "BUGSExamples")
probe = """
    using JuliaBUGS
    E = JuliaBUGS.BUGSExamples
    println("xbar=", E.rats.data.xbar, " circle_ref=", E.circle.reference_results !== nothing, " toy=", haskey(E.VOLUME_2, :toy))
    """
show_state(label) = println(rpad(label, 36), readchomp(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $probe`))

println("Julia ", VERSION)
show_state("before")

path = joinpath(root, "volume_1", "rats", "data.json")
write(path, replace(read(path, String), "\"xbar\": 22" => "\"xbar\": 23"))
show_state("after editing rats data.json")

write(joinpath(root, "volume_3", "circle", "reference.json"), """{"x": {"mean": 0.0, "std": 1.0}}""")
show_state("after adding circle reference.json")

mkpath(joinpath(root, "volume_2", "toy"))
write(joinpath(root, "volume_2", "toy", "example.toml"), "name = \"Toy\"\n")
write(joinpath(root, "volume_2", "toy", "model.bugs"), "model {\n  x ~ dnorm(0, 1)\n}\n")
show_state("after adding volume_2/toy")
