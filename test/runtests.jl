using Test, Distributed, SharedArrays
using Images, JLD
using RegisterDriver, RegisterWorkerShell

driverprocs = addprocs(2)
include("WorkerDummy.jl")
@sync for p in driverprocs
    @spawnat p eval(quote
        using Pkg
        Pkg.activate(".")
        Pkg.instantiate()
        include("WorkerDummy.jl")
    end)
end
using .WorkerDummy

workdir = tempname()
mkdir(workdir)

img = AxisArray(SharedArray{Float32}((100,100,7)), :y, :x, :time)

# Single-process tests
# Simple operation & passing back scalars
alg = Alg1(rand(3,3), 3.2)
mon = monitor(alg, (:λ,))
fn = joinpath(workdir, "file1.jld")
driver(fn, alg, img, mon)
λ = JLD.load(fn, "λ")
@test λ == Float64[1,2,3,4,5,6,7]
rm(fn)

# Passing back arrays
alg = Alg2(rand(100,100), Float32, (3,3))
mon = monitor(alg, (:tform,:u0))
fn = joinpath(workdir, "file2.jld")
driver(fn, alg, img, mon)
tform = JLD.load(fn, "tform")
u0    = JLD.load(fn, "u0")
@test tform[:,4] == collect(range(1, stop=12, length=12).+4)
@test u0[:,:,2] == fill(-2,(3,3))
rm(fn)

# Passing back strings. Anything not "packable" ends up in a group,
# one per stack.
alg = Alg3("Hello")
mon = monitor(alg, (:string,))
mon[:extra] = ""
fn = joinpath(workdir, "file3.jld")
driver(fn, alg, img, mon)
jldopen(fn) do file
    g = file["stack5"]
    @test read(g, "string") == "Hello"
    @test read(g, "extra")  == "world"
end
rm(fn)

# Multi-process
nw = length(driverprocs)
alg = Vector{Any}(undef, nw)
mon = Vector{Any}(undef, nw)
for i = 1:nw
    alg[i] = Alg2(rand(100,100), Float32, (3,3), pid=driverprocs[i])
    mon[i] = monitor(alg[i], (:tform,:u0,:workerpid))
end
fn = joinpath(workdir, "file4.jld")
driver(fn, alg, img, mon)
wpid = JLD.load(fn, "workerpid")
indx = unique(indexin(wpid, driverprocs))
@test length(indx) == length(driverprocs) && all(indx .> 0)

rmprocs(driverprocs, waitfor=1.0)
