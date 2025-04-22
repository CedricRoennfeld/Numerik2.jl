### helper.jl — docstrings, examples, comments, and style-compliant formatting

"""
    norm(x::Number, p::Real=2)

Convenience method that treats a scalar as a 1-element vector for computing norms.
Calls `norm([x], p)` internally.

# Examples
```jldoctest
julia> norm(3)
3.0

julia> norm(-2.0)
2.0

julia> norm(4.0, 0)
1.0
```
"""
norm(x::Number, p::Real=2) = norm([x], p)

"""
    norm(itr::AbstractVector{T}, p::Real=2) where T

Computes the `p`-norm of a vector-like iterable `itr`.

Supports various special cases:
- `p = 2` (default): Euclidean norm
- `p = 1`: Manhattan norm
- `p = Inf`: maximum norm
- `p = -Inf`: minimum norm
- `p = 0`: number of nonzero entries
- Arbitrary real values for general `p`

# Examples
```jldoctest
julia> norm([1, 4, -8])
9.0

julia> norm([1, 4, -8], 1)
13.0

julia> norm([1, 4, -8], Inf)
8.0

julia> norm([1, 4, -8], -Inf)
1.0

julia> norm([1, 4, -8], 0)
3.0

julia> norm([1, 4, -8], -2)
0.9630868246861536
```
"""
function norm(itr, p::Real=2)
    if p == Inf
        return float(maximum(abs, itr))
    elseif p == -Inf
        return float(minimum(abs, itr))
    elseif p == 0
        return float(count(!iszero, itr))
    elseif p == 2
        return sqrt(sum(x -> (float(x))^2, itr))
    else
        return sum(x -> abs(float(x))^p, itr)^(1/p)
    end
end

"""
    dot(x::AbstractVector{T}, y::AbstractVector{T}) where T

Computes the dot product of two vectors `x` and `y`.

# Examples
```jldoctest
julia> dot([1, 2], [3, 4])
11
```
"""
function dot(x::AbstractVector, y::AbstractVector)
    T = promote_type(eltype(x), eltype(y))
    return sum(conj(x) .* y; init = zero(T))
end

# --- Swap / Utility functions ---

"""
    swapcols!(A, i, j, domain=:)

Swaps columns `i` and `j` of matrix `A`. If `A` is a vector, swaps elements `i` and `j`.

An optional `domain` argument allows swapping over a subrange.
"""
function swapcols!(A::Union{AbstractMatrix{T}, AbstractVector{T}}, 
                   i::Integer, j::Integer, domain = :) where T
    A[[j, i], domain] = A[[i, j], domain]
end

"""
    swaprows!(A, i, j, domain=:)

Swaps rows `i` and `j` of matrix `A`.
"""
function swaprows!(A::Union{AbstractMatrix{T}, AbstractVector{T}}, 
                   i::Integer, j::Integer, domain = :) where T
    A[domain, [j, i]] = A[domain, [i, j]]
end

"""
    swapvec!(A, i, j)

Swaps two entries `i` and `j` in a vector `A`.

Alias for `swapcols!` when used with vectors.
"""
function swapvec!(A::AbstractVector{T}, i::Integer, j::Integer) where T
    swapcols!(A, i, j)
end

# --- Matrix part extractors ---

"""
    tril(A)

Returns the lower triangular part of matrix `A`, zeroing all elements above the diagonal.
"""
function tril(A::AbstractMatrix{T}) where T
    return [i >= j ? A[i, j] : zero(eltype(A)) for i in axes(A, 1), j in axes(A, 2)]
end

"""
    triu(A)

Returns the upper triangular part of matrix `A`, zeroing all elements below the diagonal.
"""
function triu(A::AbstractMatrix{T}) where T
    return [i <= j ? A[i, j] : zero(eltype(A)) for i in axes(A, 1), j in axes(A, 2)]
end

# --- Structure checkers ---

issquare(A::AbstractMatrix) = size(A, 1) == size(A, 2)

issymmetric(A::AbstractMatrix) = A == transpose(A)

ishermitian(A::AbstractMatrix) = A == A'

function istril(A::AbstractMatrix{T}) where T<:Number
    for i in axes(A, 1), j in i+1:size(A, 2)
        if A[i, j] != zero(eltype(A))
            return false
        end
    end
    return true
end

function istriu(A::AbstractMatrix{T}) where T<:Number
    for i in axes(A, 1), j in 1:i-1
        if A[i, j] != zero(eltype(A))
            return false
        end
    end
    return true
end
