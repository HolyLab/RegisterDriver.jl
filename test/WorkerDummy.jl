### Dummy algorithms to test features of `driver`
module WorkerDummy

using RegisterWorkerShell, Distributed
import RegisterWorkerShell: worker

export Alg1, Alg2, Alg3, Alg4

# Dispatch on the algorithm used to perform registration
# Each algorithm has a container it uses for storage and communication
# with the driver process
abstract type Alg <: AbstractWorker end

mutable struct Alg1{A <: AbstractArray} <: Alg
    fixed::A
    λ::Float64
    workertid::Int
end
function Alg1(fixed, λ; tid = 1)
    return Alg1(fixed, λ, tid)
end

mutable struct Alg2{A <: AbstractArray, V <: AbstractVector, M <: AbstractMatrix} <: Alg
    fixed::A
    tform::V
    u0::M
    workertid::Int
end
function Alg2(fixed, ::Type{T}, sz; tid = 1) where {T}
    return Alg2(fixed, Vector{T}(undef, 12), Matrix{T}(undef, sz), tid)
end

mutable struct Alg3 <: Alg
    string::String
    workertid::Int
end
function Alg3(s::String; tid = 1)
    return Alg3(s, tid)
end

# Here are the "registration algorithms"
function worker(algorithm::Alg1, moving, tindex, mon)
    algorithm.λ = tindex
    return monitor!(mon, algorithm)   # just dump output
end

function worker(algorithm::Alg2, moving, tindex, mon)
    # Do stuff to set tform
    tform = range(1, stop = 12, length = 12) .+ tindex
    monitor!(mon, :tform, tform)
    # Do more computations...
    return monitor!(mon, :u0, zeros(size(algorithm.u0)) .- tindex)
end

function worker(algorithm::Alg3, moving, tindex, mon)
    monitor!(mon, algorithm)
    if haskey(mon, :extra)
        mon[:extra] = "world"
    end
    return mon
end

# Alg4: monitor contains a non-BitsType array (Vector{ComplexF32}) alongside
# an unpackable string, exercising the group-write paths in the driver and initialize_jld!
mutable struct Alg4 <: Alg
    data::Vector{ComplexF32}
    label::String
    workertid::Int
end
function Alg4(; tid=1)
    return Alg4(ComplexF32[ComplexF32(float(i), -float(i)) for i in 1:4], "frame", tid)
end

function worker(algorithm::Alg4, moving, tindex, mon)
    mon[:data] = algorithm.data .* tindex
    mon[:label] = algorithm.label * string(tindex)
    return mon
end

end  # module
