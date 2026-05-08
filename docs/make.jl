using Documenter
using RegisterDriver

DocMeta.setdocmeta!(RegisterDriver, :DocTestSetup, :(using RegisterDriver); recursive=true)

makedocs(;
    modules=[RegisterDriver],
    sitename="RegisterDriver.jl",
    checkdocs=:exports,
    format=Documenter.HTML(;
        canonical="https://holylab.github.io/RegisterDriver.jl",
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/HolyLab/RegisterDriver.jl",
    devbranch="master",
)
