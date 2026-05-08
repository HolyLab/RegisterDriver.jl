using Test, Distributed, SharedArrays
using Aqua
using Documenter
using ExplicitImports
using ImageCore, JLD
using RegisterDriver, RegisterWorkerShell
using AxisArrays: AxisArray
using RegisterCore: NumDenom
using StaticArrays: SVector
using Base.Threads

push!(LOAD_PATH, pwd())
using WorkerDummy

@testset "Doctests" begin
    DocMeta.setdocmeta!(RegisterDriver, :DocTestSetup, :(using RegisterDriver); recursive = true)
    doctest(RegisterDriver; manual = false)
end

@testset "Aqua" begin
    Aqua.test_all(RegisterDriver)
end

@testset "ExplicitImports" begin
    # BitsType is intentionally accessed as HDF5.BitsType (non-public alias)
    ExplicitImports.test_explicit_imports(RegisterDriver; ignore=(:BitsType,))
end

@testset "RegisterDriver" begin
    workdir = tempname()
    mkdir(workdir)

    img = AxisArray(SharedArray{Float32}((100, 100, 7)), :y, :x, :time)

    # Single-process tests : mon::Dict
    # Simple operation & passing back scalars
    alg = Alg1(rand(3, 3), 3.2)
    mon = monitor(alg, (:λ,))
    fn = joinpath(workdir, "file1.jld")
    driver(fn, alg, img, mon)
    λ = JLD.load(fn, "λ")
    @test λ == Float64[1, 2, 3, 4, 5, 6, 7]
    rm(fn)

    # Passing back arrays
    alg = Alg2(rand(100, 100), Float32, (3, 3))
    mon = monitor(alg, (:tform, :u0))
    fn = joinpath(workdir, "file2.jld")
    driver(fn, alg, img, mon)
    tform = JLD.load(fn, "tform")
    u0 = JLD.load(fn, "u0")
    @test tform[:, 4] == collect(range(1, stop = 12, length = 12) .+ 4)
    @test u0[:, :, 2] == fill(-2, (3, 3))
    rm(fn)

    # Passing back strings. Anything not "packable" ends up in a group,
    # one per stack.
    alg = Alg3("Hello")
    mon = monitor(alg, (:string,))
    @show typeof(mon)
    mon[:extra] = ""
    fn = joinpath(workdir, "file3.jld")
    driver(fn, alg, img, mon)
    jldopen(fn) do file
        g = file["stack5"]
        @test read(g, "string") == "Hello"
        @test read(g, "extra") == "world"
    end
    rm(fn)

    # Multi-thread : mon::Vector{Dict}
    tids = threadids()
    nt = length(tids)
    alg = Vector{Any}(undef, nt)
    mon = Vector{Any}(undef, nt)
    for i in 1:nt
        alg[i] = Alg2(rand(100, 100), Float32, (3, 3), tid = tids[i])
        mon[i] = monitor(alg[i], (:tform, :u0, :workertid))
    end
    fn = joinpath(workdir, "file4.jld")
    driver(fn, alg, img, mon)
    tform = JLD.load(fn, "tform")
    u0 = JLD.load(fn, "u0")
    @test tform[:, 4] == collect(range(1, stop = 12, length = 12) .+ 4)
    @test u0[:, :, 2] == fill(-2, (3, 3))
    tid = JLD.load(fn, "workertid")
    indx = unique(indexin(tid, tids))
    @test length(indx) == length(tids) && all(indx .> 0)
    rm(fn)

    # Non-BitsType array (ComplexF32) alongside an unpackable string: exercises
    # the group-write path in the writer task (line 98) and initialize_jld! (line 169)
    alg4 = Alg4()
    mon4 = Dict{Symbol,Any}(:data => copy(alg4.data), :label => alg4.label)
    fn = joinpath(workdir, "file5.jld")
    driver(fn, alg4, img, mon4)
    jldopen(fn, "r") do file
        g2 = file["stack2"]
        @test read(g2, "label") == "frame2"
        @test read(g2, "data") == alg4.data .* 2
    end
    rm(fn)
end

@testset "In-memory single-image driver" begin
    single_img = AxisArray(SharedArray{Float32}((100, 100, 1)), :y, :x, :time)
    alg = Alg1(rand(3, 3), 3.2)
    mon = monitor(alg, (:λ,))
    result = driver(alg, single_img, mon)
    @test result[:λ] == 1.0

    multi_img = AxisArray(SharedArray{Float32}((100, 100, 3)), :y, :x, :time)
    @test_throws "With multiple images" driver(alg, multi_img, mon)
end

@testset "nicehdf5 specializations" begin
    # Plain SharedArray → sdata
    sa = SharedArray{Float32}((3, 4))
    sa .= 2.0f0
    @test RegisterDriver.nicehdf5(sa) === sdata(sa)

    # Array of StaticArrays → reinterpreted matrix
    arr_sv = [SVector(1.0, 2.0), SVector(3.0, 4.0)]
    result_sv = RegisterDriver.nicehdf5(arr_sv)
    @test result_sv == [1.0 3.0; 2.0 4.0]

    # Array of NumDenom → reinterpreted 2×n matrix
    arr_nd = [NumDenom(1.0, 2.0), NumDenom(3.0, 4.0)]
    result_nd = RegisterDriver.nicehdf5(arr_nd)
    @test result_nd == [1.0 3.0; 2.0 4.0]
end
