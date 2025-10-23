module RegisterDriver

using ImageCore, ImageMetadata, JLD, HDF5, StaticArrays, Formatting, SharedArrays, Distributed
using RegisterCore, RegisterWorkerShell
using Base.Threads

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

export driver, mm_package_loader, threadids

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
function driver(outfile::AbstractString, algorithms::Vector, img, mon::Vector)
    nalgs = length(algorithms)
    nummon = length(mon)
    nummon == nalgs || error("Number of monitors must equal number of workers")
    usethreads = nummon > 2
    numthreads = nthreads()
    tpool = map(alg->alg.workertid, algorithms)
    aindices = usethreads ? Dict(map((alg,aidx)->(alg.workertid=>aidx), algorithms, 1:length(algorithms))...) :
                            Dict(threadid()=>1)
    n = nimages(img)
    fs = FormatSpec("0$(ndigits(n))d")

    println("Initializing algorithm")
    init!(algorithms[1])

    println("Working on algorithm and saving the result")
    jldopen(outfile, "w") do file
        dsets = Dict{Symbol,Any}()
        firstsave = Ref(true)
        have_unpackable = Ref(false)

        # Channel for passing results from threads to writer
        results_ch = Channel{Tuple{Int,Dict}}(32)

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
                for (k,v) in monres
                    if isa(v, Number)
                        dsets[k][movidx] = v
                    elseif isa(v, Array) || isa(v, SharedArray)
                        vw = nicehdf5(v)
                        if eltype(vw) <: BitsType
                            colons = [Colon() for _=1:ndims(vw)]
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

        if usethreads
            # writer_task shares the first thread, making static scheduling inefficient
            @threads :dynamic for movidx in 1:n
                tid = threadid()
                if tid in tpool
                    println("thread $tid processing $movidx")
                    tmp = worker(algorithms[aindices[tid]], img, movidx, mon[aindices[tid]])
                    put!(results_ch, (movidx, deepcopy(tmp)))
                end
                yield()
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

    println("Closing algorithm")
    close!(algorithms[1])

    return nothing
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

mm_package_loader(algorithms::Vector{W}) where {W<:AbstractWorker} = mm_package_loader(algorithms[1])
function mm_package_loader(algorithm::AbstractWorker)
    load_mm_package(algorithm.dev)
    nothing
end

function threadids()
    nt = nthreads()
    ch = Channel{Int}(nt*1001)

    @threads for i in 1:nt
        put!(ch, threadid())
    end
    @sync for i in 1:(nt*1000)
        Threads.@spawn put!(ch, threadid())
    end

    close(ch)
    tids = unique(collect(ch))
    sort(tids)
end
end # module
