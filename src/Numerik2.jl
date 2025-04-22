# src/Numerik2.jl
module Numerik2

include("decomp.jl")
include("helper.jl")
include("initializers.jl")
include("solver.jl")

export 
# Functions
    lu,
    qr,
    qr_householder!,
    qr_householder,
    apply_Qt!,
    apply_Qt,
    build_Q,
    chol,
    norm,

    norm,
    dot,
    swapcols!,
    swaprows!,
    swapvec!,
    tril,
    triu,
    issquare,
    issymmetric,
    ishermitian,
    istril,
    istriu,

    eye,

    jacobi,
    iterative_solver,
    linear_solve

end # Module