### Dummy algorithms to test features of `driver`
module WorkerDummy

using RegisterWorkerShell, Distributed
import RegisterWorkerShell: worker

export Alg1, Alg2, Alg3

# Dispatch on the algorithm used to perform registration
# Each algorithm has a container it uses for storage and communication
# with the driver process
abstract type Alg <: AbstractWorker end

mutable struct Alg1{A<:AbstractArray} <: Alg
    fixed::A
    λ::Float64
    workerpid::Int
end
function Alg1(fixed, λ; pid=1)
    Alg1(maybe_sharedarray(fixed, pid), λ, pid)
end

mutable struct Alg2{A<:AbstractArray,V<:AbstractVector,M<:AbstractMatrix} <: Alg
    fixed::A
    tform::V
    u0::M
    workerpid::Int
end
function Alg2(fixed, ::Type{T}, sz; pid=1) where T
    Alg2(maybe_sharedarray(fixed, pid), maybe_sharedarray(T, (12,), pid), maybe_sharedarray(T, sz, pid), pid)
end

mutable struct Alg3 <: Alg
    string::String
    workerpid::Int
end
function Alg3(s::String; pid=1)
    Alg3(s, pid)
end

# Here are the "registration algorithms"
function worker(algorithm::Alg1, moving, tindex, mon)
    algorithm.λ = tindex
    monitor!(mon, algorithm)   # just dump output
end

function worker(algorithm::Alg2, moving, tindex, mon)
    # Do stuff to set tform
    tform = range(1, stop=12, length=12).+tindex
    monitor!(mon, :tform, tform)
    # Do more computations...
    monitor!(mon, :u0, zeros(size(algorithm.u0)).-tindex)
end

function worker(algorithm::Alg3, moving, tindex, mon)
    monitor!(mon, algorithm)
    if haskey(mon, :extra)
        mon[:extra] = "world"
    end
    mon
end

end  # module
