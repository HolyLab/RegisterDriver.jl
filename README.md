# RegisterDriver.jl

[![CI](https://github.com/HolyLab/RegisterDriver.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/HolyLab/RegisterDriver.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/HolyLab/RegisterDriver.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/HolyLab/RegisterDriver.jl)
[![docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://holylab.github.io/RegisterDriver.jl/stable)
[![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://holylab.github.io/RegisterDriver.jl/dev)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

RegisterDriver.jl drives image registration workflows in the
[BlockRegistration](https://github.com/HolyLab/BlockRegistration.jl) ecosystem.
It runs `AbstractWorker` algorithms frame-by-frame across an image stack,
optionally in parallel across multiple threads, and saves results to disk in JLD
format.

## Installation

RegisterDriver.jl is distributed through the
[HolyLab registry](https://github.com/HolyLab/HolyLabRegistry).
Add that registry before installing:

```julia
using Pkg
pkg"registry add https://github.com/HolyLab/HolyLabRegistry.git"
Pkg.add("RegisterDriver")
```

## Concepts

### Workers and monitors

A *worker* is an `AbstractWorker` instance (from a `RegisterWorker*` package
such as
[RegisterWorkerApertures](https://github.com/HolyLab/RegisterWorkerApertures.jl))
that encapsulates a registration algorithm for a particular compute device.
Before running, create a *monitor* dict that names which computed quantities to
collect from each frame:

```julia
algorithm = MyWorker(fixed, params...)         # construct an AbstractWorker
mon = monitor(algorithm, (:tform, :mismatch))  # fields to record
```

### The driver

`driver` iterates the worker over every frame of an image stack. It handles
initialisation, per-frame execution, and teardown, then either saves the
collected values to a JLD file or returns them in-memory for single-image use.

## Usage

### Single image (in-memory)

```julia
result = driver(algorithm, img, mon)
tform  = result[:tform]
```

### Image stack saved to a file

```julia
driver("results.jld", algorithm, img, mon)
```

Results for each frame are stored inside the JLD file: scalars as plain
vectors, bit-type arrays as higher-dimensional HDF5 datasets, and other values
inside per-frame `"stack<n>"` groups.

### Parallel multi-threaded registration

Start Julia with multiple threads (e.g. `julia --threads=4`), then assign one
worker per worker thread:

```julia
tids       = threadids()
algorithms = [MyWorker(fixed, params...; workertid=t) for t in tids]
monitors   = [monitor(algorithms[1], (:tform, :mismatch)) for _ in tids]
driver("results.jld", algorithms, monitors)
```

`threadids()` returns the sorted list of thread IDs that Julia actually
schedules `@threads` tasks on (typically excluding thread 1, which drives the
writer).

### Loading a device-specific backend

Some workers require a device-specific mismatch package (e.g. a CUDA backend)
to be loaded on the driver process before registration starts:

```julia
prepare_mm_package(algorithm)
driver("results.jld", algorithm, img, mon)
```

For a full introduction see the
[RegisterWorkerApertures documentation](https://github.com/HolyLab/RegisterWorkerApertures.jl).
