module RegisterDriver

using ImageCore, ImageMetadata, JLD, HDF5, StaticArrays, Formatting, SharedArrays, Distributed
using RegisterCore
using RegisterWorkerShell

if isdefined(HDF5, :BitsType)
    const BitsType = HDF5.BitsType
else
    const BitsType = HDF5.HDF5BitsKind
end
if !isdefined(HDF5, :create_dataset)
    const create_dataset = d_create
end
if !isdefined(HDF5, :create_group)
    const create_group = g_create
end

export driver, mm_package_loader

"""
`driver(outfile, algorithm, img, mon)` performs registration of the
image(s) in `img` according to the algorithm selected by
`algorithm`. `algorithm` is either a single instance, or for parallel
computation a vector of instances, of `AbstractWorker` types.  See the
`RegisterWorkerShell` module for more details.

Results are saved in `outfile` according to the information in `mon`.
`mon` is a `Dict`, or for parallel computation a vector of `Dicts` of
the same length as `algorithm`.  The data saved correspond to the keys
(always `Symbol`s) in `mon`, and the values are used for communication
between the worker(s) and the driver.  The usual way to set up `mon`
is like this:

```
    algorithm = RegisterRigid(fixed, params...)   # An AbstractWorker algorithm
    mon = monitor(algorithm, (:tform,:mismatch))  # List of variables to record
```

The list of symbols, taken from the field names of `RegisterRigid`,
specifies the pieces of information to be communicated back to the
driver process for saving and/or display to the user.  It's also
possible to request local variables in the worker, as long as the
worker has been written to look for such settings:

```
    # <in the worker algorithm>
    monitor_copy!(mon, :extra, extra)
```

which will save `extra` only if `:extra` is a key in `mon`.
"""
function driver(outfile::AbstractString, algorithm::Vector, img, mon::Vector)
    nworkers = length(algorithm)
    length(mon) == nworkers || error("Number of monitors must equal number of workers")
    use_workerprocs = nworkers > 1 || workerpid(algorithm[1]) != myid()
    rralgorithm = Array{RemoteChannel}(undef, nworkers)
    if use_workerprocs
        # Push the algorithm objects to the worker processes. This elminates
        # per-iteration serialization penalties, and ensures that any
        # initalization state is retained.
        for i = 1:nworkers
            alg = algorithm[i]
            rralgorithm[i] = put!(RemoteChannel(workerpid(alg)), alg)
        end
        # Perform any needed worker initialization
        @sync for i = 1:nworkers
            p = workerpid(algorithm[i])
            @async remotecall_fetch(init!, p, rralgorithm[i])
        end
    else
        init!(algorithm[1])
    end
    try
        n = nimages(img)
        fs = FormatSpec("0$(ndigits(n))d")  # group names of unpackable objects
        jldopen(outfile, "w") do file
            dsets = Dict{Symbol,Any}()
            firstsave = SharedArray{Bool}(1)
            firstsave[1] = true
            have_unpackable = SharedArray{Bool}(1)
            have_unpackable[1] = false
            # Run the jobs
            nextidx = 0
            getnextidx() = nextidx += 1
            writing_mutex = RemoteChannel()
            @sync begin
                for i = 1:nworkers
                    alg = algorithm[i]
                    @async begin
                        while (idx = getnextidx()) <= n
                            if use_workerprocs
                                remotecall_fetch(println, workerpid(alg), "Worker ", workerpid(alg), " is working on ", idx)
                                # See https://github.com/JuliaLang/julia/issues/22139
                                tmp = remotecall_fetch(worker, workerpid(alg), rralgorithm[i], img, idx, mon[i])
                                copy_all_but_shared!(mon[i], tmp)
                            else
                                println("Working on ", idx)
                                mon[1] = worker(algorithm[1], img, idx, mon[1])
                            end
                            # Save the results
                            put!(writing_mutex, true)  # grab the lock
                            try
                                local g
                                if firstsave[]
                                    firstsave[] = false
                                    have_unpackable[] = initialize_jld!(dsets, file, mon[i], fs, n)
                                end
                                if fetch(have_unpackable[])
                                    g = file[string("stack", fmt(fs, idx))]
                                end
                                for (k,v) in mon[i]
                                    if isa(v, Number)
                                        dsets[k][idx] = v
                                        continue
                                    elseif isa(v, Array) || isa(v, SharedArray)
                                        vw = nicehdf5(v)
                                        if eltype(vw) <: BitsType
                                            colons = [Colon() for i = 1:ndims(vw)]
                                            dsets[k][colons..., idx] = vw
                                            continue
                                        end
                                    end
                                    g[string(k)] = v
                                end
                            finally
                                take!(writing_mutex)   # release the lock
                            end
                        end
                    end
                end
            end
        end
    finally
        # Perform any needed worker cleanup
        if use_workerprocs
            @sync for i = 1:nworkers
                p = workerpid(algorithm[i])
                @async remotecall_fetch(close!, p, rralgorithm[i])
            end
        else
            close!(algorithm[1])
        end
    end
end

driver(outfile::AbstractString, algorithm::AbstractWorker, img, mon::Dict) = driver(outfile, [algorithm], img, [mon])

"""
`mon = driver(algorithm, img, mon)` performs registration on a single
image, returning the results in `mon`.
"""
function driver(algorithm::AbstractWorker, img, mon::Dict)
    nimages(img) == 1 || error("With multiple images, you must store results to a file")
    init!(algorithm)
    worker(algorithm, img, 1, mon)
    close!(algorithm)
    mon
end

# Initialize the datasets in the output JLD file.
# We wait to do this until we get back one valid `mon` object,
# to get the sizes of any returned arrays.
function initialize_jld!(dsets, file, mon, fs, n)
    have_unpackable = false
    for (k,v) in mon
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
        for i = 1:n
            create_group(file, string("stack", fmt(fs, i)))
        end
    end
    have_unpackable
end

function nicehdf5(v::Union{Array{T},SharedArray{T}}) where T<:StaticArray
    nicehdf5(reshape(reinterpret(eltype(T), vec(sdata(v))), (size(eltype(v))..., size(v)...)))
end

function nicehdf5(v::Union{Array{T},SharedArray{T}}) where T<:NumDenom
    nicehdf5(reshape(reinterpret(eltype(T), vec(sdata(v))), (2, size(v)...)))
end

nicehdf5(v::SharedArray) = sdata(v)
nicehdf5(v) = v

function copy_all_but_shared!(dest, src)
    for (k, v) in src
        if !isa(v, SharedArray)
            dest[k] = v
        end
    end
    dest
end

mm_package_loader(algorithm::AbstractWorker) = mm_package_loader([algorithm])
function mm_package_loader(algorithms::Vector)
    nworkers = length(algorithms)
    use_workerprocs = nworkers > 1 || workerpid(algorithms[1]) != myid()
    rrdev = Array{RemoteChannel}(undef, nworkers)
    if use_workerprocs
        for i = 1:nworkers
            dev = algorithms[i].dev
            rrdev[i] = put!(RemoteChannel(workerpid(algorithms[i])), dev)
        end
        @sync for i = 1:nworkers
            p = workerpid(algorithms[i])
            @async remotecall_fetch(load_mm_package, p, rrdev[i])
        end
    else
        load_mm_package(algorithms[1].dev)
    end
    nothing
end

end # module
