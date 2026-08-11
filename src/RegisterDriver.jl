"""
    RegisterDriver

Drive image registration workflows: run `AbstractWorker` algorithms over
single- or multi-threaded execution and save results to disk.

Primary entry point: [`driver`](@ref). See also [`prepare_mm_package`](@ref)
and [`threadids`](@ref).
"""
module RegisterDriver

using Distributed: Distributed
using Formatting: Formatting, FormatSpec, fmt
using HDF5: HDF5, create_dataset, create_group, dataspace, datatype
using ImageCore: ImageCore, nimages
using ImageMetadata: ImageMetadata
using JLD: JLD, jldopen
using RegisterCore: RegisterCore, NumDenom
using RegisterWorkerShell: RegisterWorkerShell, AbstractWorker, ArrayDecl,
                           close!, init!, load_mm_package, worker
using SharedArrays: SharedArrays, SharedArray, sdata
using StaticArrays: StaticArrays, StaticArray
using Base.Threads: @threads, nthreads, threadid

const BitsType = HDF5.BitsType

export driver, prepare_mm_package, threadids

"""
    driver(outfile, algorithm, img, mon)
    driver(outfile, algorithms, img, mon)

Register the image(s) in `img` and save results to `outfile` in JLD format.

`algorithm` is a single `AbstractWorker` instance; `algorithms` is an `AbstractVector`
of such instances for parallel (multi-threaded) computation. See the
`RegisterWorkerShell` module for details on constructing workers.

`mon` is an `AbstractDict` mapping `Symbol` keys to communication values, or for the
parallel form an `AbstractVector` of such `AbstractDict`s (one per worker). The keys specify
which computed quantities are communicated back from each worker. Set them up
with the worker's `monitor` function:

```julia
algorithm = RegisterRigid(fixed, params...)     # construct an AbstractWorker
mon = monitor(algorithm, (:tform, :mismatch))   # select fields to record
driver("results.jld", algorithm, img, mon)      # register and save
```

Scalars are stored as plain vectors indexed by image number; bit-type arrays are
stored as higher-dimensional HDF5 datasets; other values are stored per-image
inside `"stack<n>"` groups.

Additional local worker variables can be recorded by adding their keys to `mon`
and calling `monitor_copy!` inside the worker:

```julia
# inside the worker algorithm:
monitor_copy!(mon, :extra, extra)   # saved only if :extra is a key in mon
```

Pass `parallel=true` (or `false`) to force multi-threaded (or sequential)
execution. By default, `parallel = length(algorithms) > 2`.

Returns `nothing`.
"""
function driver(outfile::AbstractString, algorithms::AbstractVector, img, mon::AbstractVector;
                parallel::Bool = length(algorithms) > 2)
    nalgs = length(algorithms)
    nummon = length(mon)
    nummon == nalgs || error("Number of monitors must equal number of workers")
    n = nimages(img)
    fs = FormatSpec("0$(ndigits(n))d")

    println("Initializing algorithms")
    # Every worker is initialized, not just the first: `init!` sets up per-worker
    # resources (device contexts, scratch buffers), and every worker registers
    # images of its own.
    foreach(init!, algorithms)

    println("Working on algorithm and saving the result")
    jldopen(outfile, "w") do file
        dsets = Dict{Symbol, Any}()
        firstsave = Ref(true)
        have_unpackable = Ref(false)

        # Channel for passing results from threads to writer
        results_ch = Channel{Tuple{Int, Dict}}(32)

        # Writer task (runs on main thread)
        writer_task = @async begin
            for (movidx, monres) in results_ch

                # Initialize datasets on first save
                if firstsave[]
                    firstsave[] = false
                    have_unpackable[] = initialize_jld!(dsets, file, monres, fs, n)
                end

                g = have_unpackable[] ? file[string("stack", fmt(fs, movidx))] : nothing

                # Write all values into the file
                for (k, v) in monres
                    if isa(v, Number)
                        dsets[k][movidx] = v
                    elseif isa(v, Array) || isa(v, SharedArray)
                        vw = nicehdf5(v)
                        if eltype(vw) <: BitsType
                            colons = [Colon() for _ in 1:ndims(vw)]
                            dsets[k][colons..., movidx] = vw
                        else
                            g[string(k)] = v
                        end
                    else
                        g[string(k)] = v
                    end
                end
                yield()
            end
        end

        if parallel
            # One task per worker, each taking the next unclaimed image. A worker
            # must not be chosen by `threadid()`: tasks spawned per image can
            # interleave on a thread at any yield point, so two running
            # concurrently may observe the same id and would then share one
            # worker — and one monitor dict — corrupting both registrations.
            # Owning a worker for the task's whole life makes that impossible
            # without any locking.
            nextidx = Threads.Atomic{Int}(1)
            @sync for k in eachindex(algorithms, mon)
                Threads.@spawn while true
                    movidx = Threads.atomic_add!(nextidx, 1)
                    movidx > n && break
                    println("worker $k processing $movidx")
                    # `mon[k]` is reused for every image this worker handles, so
                    # the writer needs a snapshot rather than a live reference.
                    tmp = worker(algorithms[k], img, movidx, mon[k])
                    put!(results_ch, (movidx, deepcopy(tmp)))
                end
            end
        else
            for movidx in 1:n
                println("processing $movidx")
                tmp = worker(algorithms[1], img, movidx, mon[1])
                put!(results_ch, (movidx, tmp))
                yield()
            end
        end

        # Close channel and wait for writer to finish
        close(results_ch)
        wait(writer_task)
    end

    println("Closing algorithms")
    foreach(close!, algorithms)

    return nothing
end

driver(outfile::AbstractString, algorithm::AbstractWorker, img, mon::AbstractDict) = driver(outfile, [algorithm], img, [mon])

"""
    driver(algorithm, img, mon) -> Dict

Register the single image in `img` and return the populated result `Dict`.

`img` must contain exactly one image; for multi-image stacks use the
file-saving form of `driver`. The returned `Dict` is the same object as `mon`,
with each key's value updated to the quantity computed by the worker.

# Example

```julia
algorithm = RegisterRigid(fixed, params...)
mon = monitor(algorithm, (:tform, :mismatch))
mon = driver(algorithm, img, mon)
tform = mon[:tform]
```
"""
function driver(algorithm::AbstractWorker, img, mon::AbstractDict)
    nimages(img) == 1 || error("With multiple images, you must store results to a file")
    init!(algorithm)
    worker(algorithm, img, 1, mon)
    close!(algorithm)
    return mon
end

# Initialize the datasets in the output JLD file.
# We wait to do this until we get back one valid `mon` object,
# to get the sizes of any returned arrays.
function initialize_jld!(dsets, file, mon, fs, n)
    have_unpackable = false
    for (k, v) in mon
        kstr = string(k)
        if isa(v, Number)
            write(file, kstr, Vector{typeof(v)}(undef, n))
            dsets[k] = file[kstr]
        elseif isa(v, Array) || isa(v, SharedArray)
            v = nicehdf5(v)
            if eltype(v) <: BitsType
                fullsz = (size(v)..., n)
                dsets[k] = create_dataset(file.plain, kstr, datatype(eltype(v)), dataspace(fullsz))
            else
                write(file, kstr, Array{eltype(v)}(undef, size(v)..., n))  # might fail if it's too big, but we tried
            end
            dsets[k] = file[kstr]
        elseif isa(v, ArrayDecl)  # maybe this never happens?
            fullsz = (v.arraysize..., n)
            dsets[k] = create_dataset(file.plain, kstr, datatype(eltype(v)), dataspace(fullsz))
        else
            have_unpackable = true
        end
    end
    if have_unpackable
        for i in 1:n
            create_group(file, string("stack", fmt(fs, i)))
        end
    end
    return have_unpackable
end

function nicehdf5(v::Union{Array{T}, SharedArray{T}}) where {T <: StaticArray}
    return nicehdf5(reshape(reinterpret(eltype(T), vec(sdata(v))), (size(eltype(v))..., size(v)...)))
end

function nicehdf5(v::Union{Array{T}, SharedArray{T}}) where {T <: NumDenom}
    return nicehdf5(reshape(reinterpret(eltype(T), vec(sdata(v))), (2, size(v)...)))
end

nicehdf5(v::SharedArray) = sdata(v)
nicehdf5(v) = v


"""
    prepare_mm_package(algorithm::AbstractWorker)
    prepare_mm_package(algorithms::AbstractVector{<:AbstractWorker})

Load the mismatch-computation package appropriate for `algorithm`'s compute device.

Thin wrapper around `RegisterWorkerShell.load_mm_package` that accepts either a
single worker or a vector of workers (delegating to the first element). Call this
before `driver` when the algorithm requires a device-specific backend (e.g., a
CUDA mismatch package) to be loaded on the driver process.

Returns `nothing`.
"""
prepare_mm_package(algorithms::AbstractVector{<:AbstractWorker}) = prepare_mm_package(algorithms[1])
function prepare_mm_package(algorithm::AbstractWorker)
    load_mm_package(algorithm.dev)
    return nothing
end

"""
    threadids() -> Vector{Int}

Return the sorted list of thread IDs that Julia's scheduler actually assigns to
tasks spawned with `@threads` and `Threads.@spawn`.

Julia's main thread (ID 1) typically does not execute worker tasks.

[`driver`](@ref) does not need this: it gives each worker its own task and
distributes images between them, so `algorithms` may have any length and a
worker's `workertid` does not affect which images it receives.

# Example

```julia
# On a Julia session started with 4 threads
threadids()    # e.g. [2, 3, 4, 5]
```
"""
function threadids()
    nt = nthreads()
    ch = Channel{Int}(nt * 1001)

    @threads for i in 1:nt
        put!(ch, threadid())
    end
    @sync for i in 1:(nt * 1000)
        Threads.@spawn put!(ch, threadid())
    end

    close(ch)
    tids = unique(collect(ch))
    return sort(tids)
end
end # module
