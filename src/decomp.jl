"""
    lu(A::AbstractMatrix{T}) where T<:Number

Performs LU decomposition of a square matrix `A` using partial pivoting.

Returns a tuple `(p, L, U)` such that `A[p, :] ≈ L * U`, where:
- `p`: permutation vector (row pivoting)
- `L`: unit lower triangular matrix
- `U`: upper triangular matrix

# Errors
Throws an error if the matrix is not square or is numerically singular during pivoting.

# Examples
```jldoctest
julia> A = [4.0 3.0; 6.0 3.0]; p, L, U = lu(A);

julia> L * U ≈ A[p, :]
true
```
"""
function lu(A::AbstractMatrix{T}) where T<:Number
    if !issquare(A)
        error("LU decomposition requires a square matrix.")
    end
    n = size(A, 1)

    # Initialize identity matrix for L, copy of A for U, and permutation vector p
    L = eye(n)
    U = Float64.(A)
    p = collect(1:n)

    for k in 1:n-1
        # Partial pivoting: choose row i with max abs value in column k (from row k on)
        i = k + argmax(abs.(U[k:end, k])) - 1
        if abs(U[i, k]) < eps(Float64)
            error("LU failed: pivot ($i, $k) ≈ 0 (matrix may be singular).")
        end

        # Row swaps in U and pivot vector, adjust L if needed
        if i != k
            swapcols!(U, i, k)
            swapvec!(p, i, k)
            if k > 1
                swapcols!(L, i, k, 1:k-1)
            end
        end

        # Elimination: update rows below the pivot row
        for j in k+1:n
            L[j, k] = U[j, k] / U[k, k]
            U[j, :] -= L[j, k] * U[k, :]
        end
    end

    return p, tril(L), triu(U)
end

"""
    chol(A::AbstractMatrix{T}) where T<:Real

Performs Cholesky decomposition of a symmetric positive definite matrix `A`.

Returns lower triangular matrix `L` such that `A = L * L'`.

# Errors
Throws an error if matrix is not symmetric or not positive definite.

# Examples
```jldoctest
julia> A = [4.0 2.0; 2.0 2.25]; L = chol(A);

julia> L * L' ≈ A
true
```
"""
function chol(A::AbstractMatrix{T}) where T<:Real
    if !issymmetric(A)
        error("Matrix must be symmetric for Cholesky decomposition.")
    end

    n = size(A, 1)
    L = zeros(T, n, n)

    for i in 1:n
        for j in 1:i
            # Compute inner product of row i and column j (up to index j-1)
            s = dot(L[i, 1:j-1], L[j, 1:j-1])

            if i == j
                # Diagonal entry: ensure positive-definiteness
                diff = A[i, i] - s
                if diff <= zero(T)
                    error("Cholesky failed: diagonal at ($i, $j) is not positive.")
                end
                L[i, j] = sqrt(diff)
            else
                # Off-diagonal entry
                if L[j, j] == 0
                    error("Cholesky failed: division by zero at ($j, $j).")
                end
                L[i, j] = (A[i, j] - s) / L[j, j]
            end
        end
    end
    return L
end

"""
    qr_householder!(A::AbstractMatrix{T}) where T<:Real

In-place QR decomposition using Householder reflections.

Returns a tuple `(R, sc)` where:
- `R`: matrix with upper-triangular form and reflection vectors in lower part
- `sc`: scaling coefficients used to apply the Householder transformations

# Examples
```jldoctest
julia> A = [1.0 1.0; 1.0 -1.0]; R, sc = qr_householder!(copy(A));

julia> size(R), size(sc)
((2, 2), (2,))
```
"""
function qr_householder!(A::AbstractMatrix{T}) where T<:Real
    m, n = size(A)
    mindim = min(n, m)
    sc = zeros(T, mindim)

    for k in 1:mindim
        # Householder vector x: make column k zero below diagonal
        x = @view A[k:end, k]
        a = -sign(x[1]) * norm(x)
        x[1] -= a

        # Store scaling factor for reflector
        sc[k] = 2.0 / dot(x, x)

        # Apply transformation to remaining columns
        for j in k+1:n
            col = @view A[k:end, j]
            col .-= sc[k] * dot(x, col) * x
        end
        x[1] = a  # restore leading value for reconstruction
    end

    return A, sc
end

"""
    qr_householder(A::AbstractMatrix{T}) where T<:Real

Out-of-place version of `qr_householder!`. Returns the same result without modifying input.

# Examples
```jldoctest
julia> A = [2.0 1.0; 1.0 3.0]; R, sc = qr_householder(A);

julia> typeof(R), typeof(sc)
(Matrix{Float64}, Vector{Float64})
```
"""
function qr_householder(A::AbstractMatrix{T}) where T<:Real
    return qr_householder!(copy(A))
end

"""
    build_Q(R::AbstractMatrix{T}, sc::AbstractVector{T}) where T<:Real

Reconstructs the orthogonal matrix `Q` from Householder representation.

# Arguments
- `R`: matrix output from `qr_householder`
- `sc`: scaling vector from `qr_householder`

# Returns
An orthogonal matrix `Q` such that `A ≈ Q * R`.

# Examples
```jldoctest
julia> A = [1.0 1.0; 1.0 -1.0]; R, sc = qr_householder(A);
       Q = build_Q(R, sc);

julia> Q' * Q ≈ I
true
```
"""
function build_Q(R::AbstractMatrix{T}, sc::AbstractVector{T}) where T<:Real
    m, n = size(R)
    mindim = length(sc)
    Q = eye(T, m)

    for k = mindim:-1:1
        # Reconstruct reflection vector v from column k
        v = zeros(T, m - k + 1)
        if k < m
            v[2:end] .= R[k+1:end, k]
        end

        v_sq = 2.0 / sc[k]
        tail_sq = sum(abs2, v[2:end])
        v[1] = sqrt(v_sq - tail_sq)
        a = R[k, k]
        if a != 0
            v[1] *= -a / abs(a)
        end

        # Apply Q_k = I - s * v * v' to Q[k:end, :]
        Q_k = @view Q[k:end, :]
        Q_k .-= sc[k] * v * (v' * Q_k)
    end

    return Q
end

"""
    apply_Qt!(y::Vector{T}, R::AbstractMatrix{T}, sc::Vector{T}) where T<:Real

In-place multiplication of vector `y` by `Q'`, where `Q` is from QR decomposition.

# Returns
Updated vector `y ← Q' * y`

# Examples
```jldoctest
julia> A = [1.0 2.0; 3.0 4.0]; R, sc = qr_householder(A);

julia> b = [5.0, 6.0]; apply_Qt!(b, R, sc)
2-element Vector{Float64}:
  -6.7082
  -0.4472
```
"""
function apply_Qt!(y::Vector{T}, R::AbstractMatrix{T}, sc::Vector{T}) where T<:Real
    m = length(y)
    mindim = length(sc)

    for k = 1:mindim
        # Reconstruct Householder vector v from R
        v = zeros(T, m - k + 1)
        if k < m
            v[2:end] .= R[k+1:end, k]
        end
        v_sq = 2.0 / sc[k]
        tail_sq = sum(abs2, v[2:end])
        v[1] = sqrt(v_sq - tail_sq)
        if R[k, k] > 0
            v[1] = -v[1]
        end

        # Apply reflector: y_k ← (I - s * v * v') * y_k
        y_k = @view y[k:end]
        y_k .-= sc[k] * v * dot(v, y_k)
    end

    return y
end

"""
    apply_Qt(y::Vector{T}, R::AbstractMatrix{T}, sc::Vector{T}) where T<:Real

Out-of-place version of `apply_Qt!`. Returns `Q' * y`.

# Examples
```jldoctest
julia> A = [1.0 2.0; 3.0 4.0]; R, sc = qr_householder(A);

julia> b = [5.0, 6.0]; y = apply_Qt(b, R, sc);

julia> y
2-element Vector{Float64}:
  -6.7082
  -0.4472
```
"""
function apply_Qt(y::Vector{T}, R::AbstractMatrix{T}, sc::Vector{T}) where T<:Real
    return apply_Qt!(copy(y), R, sc)
end


"""
    qr(A::AbstractMatrix{T}) where T<:Real

Convenience function to compute full QR decomposition.

# Returns
- `Q`: orthogonal matrix
- `R`: upper-triangular matrix

# Examples
```jldoctest
julia> A = [1.0 2.0; 3.0 4.0]; Q, R = qr(A);

julia> Q * R ≈ A
true
```
""" 
function qr(A::AbstractMatrix{T}) where T<:Real 
    R, sc = qr_householder(A) 
    Q = build_Q(R, sc) 
    R_new = triu(R[1:min(size(A)...), :]) 
    return Q[:, 1:size(A, 2)], R_new 
end