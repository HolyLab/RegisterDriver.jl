### Dummy algorithms to test features of `driver`
module WorkerDummy

using RegisterWorkerShell, Distributed
import RegisterWorkerShell: worker, init!, close!

export Alg1, Alg2, Alg3, Alg4, AlgExclusive, AlgLifecycle

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

# AlgExclusive: detects a worker being used by two registrations at once.
# `driver` must never hand one worker to concurrently-running tasks, because a
# worker's fields and its monitor dict are mutated in place.
mutable struct AlgExclusive <: Alg
    busy::Bool          # set for the duration of a call, checked on entry
    reentered::Bool     # sticky: a second entry was seen while busy
    ncalls::Int
    workertid::Int
end
AlgExclusive(; tid = 1) = AlgExclusive(false, false, 0, tid)

function worker(algorithm::AlgExclusive, moving, tindex, mon)
    algorithm.busy && (algorithm.reentered = true)
    algorithm.busy = true
    algorithm.ncalls += 1
    # Yield points are what let two tasks interleave on one thread; a real
    # worker reaches them through I/O, FFT planning and the like.
    for _ in 1:20
        yield()
    end
    monitor!(mon, :tindex, tindex)
    algorithm.busy = false
    return mon
end

# AlgLifecycle: records its own `init!`/`close!` calls. A worker that registers
# images must have been initialized first, so `driver` owes every element of
# `algorithms` an `init!` and a matching `close!`.
mutable struct AlgLifecycle <: Alg
    ninit::Int
    nclose::Int
    ncalls::Int
    uninitialized::Bool   # sticky: an image arrived before `init!`
    workertid::Int
end
AlgLifecycle(; tid = 1) = AlgLifecycle(0, 0, 0, false, tid)

init!(algorithm::AlgLifecycle) = (algorithm.ninit += 1; nothing)
close!(algorithm::AlgLifecycle) = (algorithm.nclose += 1; nothing)

function worker(algorithm::AlgLifecycle, moving, tindex, mon)
    algorithm.ninit == 0 && (algorithm.uninitialized = true)
    algorithm.ncalls += 1
    monitor!(mon, :tindex, tindex)
    return mon
end

end  # module
