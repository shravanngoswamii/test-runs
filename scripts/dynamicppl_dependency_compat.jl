using Pkg

for version in ("0.15.2", "0.15.3")
    project = abspath("compat-$version")
    withenv("JULIA_PKG_PRECOMPILE_AUTO" => "0") do
        Pkg.activate(project)
        Pkg.develop(path=abspath("target"))
        Pkg.add(PackageSpec(name="AbstractPPL", version=VersionNumber(version)))
        Pkg.status(; mode=Pkg.PKGMODE_MANIFEST)
    end
    withenv("REVIEW_REVISION" => (ENV["REVIEW_HEAD"] == "true" ? "head" : "base")) do
        run(`$(Base.julia_cmd()) --project=$project scripts/dynamicppl_storage_compat.jl`)
    end
end
