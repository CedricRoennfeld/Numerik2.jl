"""
    eye(T::Type, m::Integer, n::Integer)

Returns an `m × n` identity-like matrix of type `T`, with ones on the diagonal
and zeros elsewhere. General rectangular version.

# Examples
```jldoctest
julia> eye(Float64, 2, 3)
2×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
```
"""
function eye(T::Type, m::Int, n::Int)
    I = zeros(T, m, n)
    for i in 1:min(m, n)
        I[i, i] = one(T)
    end
    return I
end

"""
    eye(n::Integer, m::Integer)

Convenience constructor for `eye(Float64, n, m)`.

# Examples
```jldoctest
julia> eye(2, 3)
2×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
```
"""
eye(n::Integer, m::Integer) = eye(Float64, n, m)

"""
    eye(T::Type, n::Integer)

Convenience constructor for `eye(T, n, n)`.

# Examples
```jldoctest
julia> eye(Int, 3)
3×3 Matrix{Int64}:
 1  0  0
 0  1  0
 0  0  1
```
"""
eye(T::Type, n::Integer) = eye(T, n, n)

"""
    eye(n::Integer)

Convenience constructor for `eye(Float64, n, n)`.

# Examples
```jldoctest
julia> eye(3)
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
```
"""
eye(n::Integer) = eye(Float64, n, n)

function diag(v::AbstractVector{T}) where T
    n = length(v)
    D = zeros(T, n, n)
    for i in axes(D,1)
        D[i,i] = v[i]
    end
    return D
end

function diag(A::AbstractMatrix{T}) where T
    n = min(size(A,1), size(A,2))
    v = zeros(T,n)
    for i in axes(v,1)
        v[i] = A[i,i]
    end
    return v
end
