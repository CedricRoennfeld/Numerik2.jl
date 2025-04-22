"""
    iterative_solver(f, x0; tol=nothing, max_iter=10000, custom_norm=x->norm(x))

Applies fixed-point iteration to `f` starting from `x0`. 

If `tol` is provided, stops when `custom_norm(x_{n+1} - x_n) < tol`, 
otherwise performs `max_iter` steps.

# Arguments
- `f`: iteration function
- `x0`: initial guess (scalar or vector)
- `tol`: desired absolute tolerance
- `max_iter`: maximum amount of iteration
- `custom_norm`: norm function (default: Euclidean norm)

# Returns:
- If tol is `nothing`: the final iterate `x`
- If tol is set: `(x, iterations)`

# Errors
Throws an error if no convergence within `max_iter` steps, but tol is set.

```jldoctest
julia> f(x) = cos(x);
       fixed_point_iteration(f, 0, 10)
0.7314040424225098

julia> fixed_point_iteration(f, 0, 1e-5)
(0.7390822985224023, 30)

julia> F(x) = [1/3 * x[2]^2 + 1/8, 1/4 * x[1]^2 - 1/6];
       fixed_point_iteration(F, [0,0], 10)
[0.13376887143948119, -0.16219313892676337]       

julia> fixed_point_iteration(F, [0, 0], 1e-30, custom_norm = x -> norm(x, Inf))
([0.13376887143813732, -0.16219313892520842], 16)
```
"""
function iterative_solver(
    f, 
    x0::Union{Number, AbstractVector};
    tol::Union{Nothing, Real} = nothing, 
    max_iter::Int = 10000, 
    custom_norm = x -> norm(x)
)
    x = copy(x0)
    if tol === nothing
        for _ in 1:max_iter
            x = f(x)
        end
        return x
    else
        for i in 1:max_iter
            x_new = f(x)
            if custom_norm(x_new - x) < tol
                return x_new, i
            end
            x = x_new
        end
        error("Iteration did not converge within $max_iter steps.")
    end
end

function jacobi(
    A::AbstractMatrix{T}, 
    b::AbstractVector{T}, 
    x0::AbstractVector{T};
    tol::Union{Nothing, Real} = nothing, 
    max_iter::Int = 10000, 
    custom_norm = x -> norm(x)
) where T<:Number
    # extract diagonal from A to exclude it in dot product
    R = A - diag(diag(A))

    f(x) = [(b[i]-dot(R[i,:],x))/A[i,i] for i in axes(x,1)]
    return iterative_solver(f, x0; tol=tol, max_iter=max_iter, custom_norm=custom_norm)
end



"""
    best_method(A::AbstractMatrix{T}; opts=NamedTuple()) where T<:Number

Returns a string representing the best linear system solving method for matrix `A`,
optionally guided by user hints passed in `opts`.

# Options
- `UT=true`: matrix is upper triangular
- `LT=true`: matrix is lower triangular
- `SPD=true`: matrix is symmetric positive definite

# Returns
One of: "triu", "tril", "spd", "lu", "qr_over", "qr_under"

# Examples
```jldoctest
julia> best_method([1.0 0; 0 1.0], UT=true)
"triu"

julia> best_method(rand(5,3))
"qr_over"
```
"""
function best_method(A::AbstractMatrix{T}; opts::NamedTuple = (;)) where T<:Number
    m, n = size(A)

    # Prefer user-provided structural hints
    if get(opts, :UT, false)
        return "triu"
    elseif get(opts, :LT, false)
        return "tril"
    elseif get(opts, :SPD, false)
        return "spd"

    # Otherwise infer based on shape
    elseif m == n
        return "lu"
    elseif m > n
        return "qr_over"
    else
        return "qr_under"
    end
end


"""
    linear_solve(A, b; method=nothing, opts=NamedTuple())

Solves the linear system `Ax = b` using an appropriate method depending on the
structure of `A`, or an explicitly provided `method`.

Supports structured hints via `opts`, including:
- `UT=true`: matrix is upper triangular
- `LT=true`: matrix is lower triangular
- `SPD=true`: matrix is symmetric positive definite

# Methods
- "triu": back substitution
- "tril": forward substitution
- "lu": LU decomposition with pivoting
- "spd": Cholesky decomposition
- "qr_over": overdetermined system via QR
- "qr_under": underdetermined system (minimum-norm solution)

# Returns
Solution vector `x` such that `Ax ≈ b`

# Errors
Throws an error if no valid method is found or method fails.

# Examples
```jldoctest
julia> A = [1.0 0; 2.0 1]; b = [1.0, 4.0];
       x = linear_solve(A, b; method="tril")

julia> A * x ≈ b
true
```
"""
function linear_solve(
    A::AbstractMatrix,
    b::AbstractVector;
    method::Union{Nothing, String} = nothing,
    opts::NamedTuple = (;)
)
    n, m = size(A)

    # If no method is provided, determine the best one using hints or shape
    if isnothing(method)
        return linear_solve(A, b; method = best_method(A; opts = opts), opts = opts)
    end

    if method == "triu"
        # Back substitution
        x = Vector{Float64}(undef, n)
        for k in n:-1:1
            x[k] = (b[k] - dot(A[k, k+1:end], x[k+1:n])) / A[k, k]
        end
        return x

    elseif method == "tril"
        # Forward substitution
        x = Vector{Float64}(undef, n)
        for k in 1:n
            x[k] = (b[k] - dot(A[k, 1:k-1], x[1:k-1])) / A[k, k]
        end
        return x

    elseif method == "lu"
        # LU decomposition with partial pivoting
        p, L, U = lu(A)
        y = linear_solve(L, b[p]; method = "tril")
        x = linear_solve(U, y; method = "triu")
        return x

    elseif method == "spd"
        # Cholesky-based solver for symmetric positive definite systems
        L = chol(A)
        y = linear_solve(L, b; method = "tril")
        x = linear_solve(L', y; method = "triu")
        return x

    elseif method == "qr_over"
        # Solve overdetermined system (least squares): A = QR
        R, sc = qr_householder(A)
        y = copy(b)
        apply_Qt!(y, R, sc)

        ncols = size(R, 2)
        y_top = y[1:ncols]
        R_top = R[1:ncols, 1:ncols]
        x = linear_solve(R_top, y_top; method = "triu")
        return x

    elseif method == "qr_under"
        # Solve underdetermined system (min-norm): A' = QR
        R, sc = qr_householder(A')
        R_t = R'
        b1 = linear_solve(R_t[1:n, 1:n], b; method = "tril")
        Q, _ = qr(A')
        x = Q * b1
        return x

    else
        error("Invalid method argument: $method")
    end
end
