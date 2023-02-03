using LinearElasticityBase
using Documenter

DocMeta.setdocmeta!(LinearElasticityBase, :DocTestSetup, :(using LinearElasticityBase); recursive=true)

makedocs(;
    modules=[LinearElasticityBase],
    authors="singularitti <singularitti@outlook.com> and contributors",
    repo="https://github.com/MineralsCloud/LinearElasticityBase.jl/blob/{commit}{path}#{line}",
    sitename="LinearElasticityBase.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MineralsCloud.github.io/LinearElasticityBase.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MineralsCloud/LinearElasticityBase.jl",
    devbranch="main",
)
