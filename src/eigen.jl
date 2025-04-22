# TODO: Eigenvectors (weren't needed yet)

"""
    hessenberg(A::AbstractMatrix{T}) where T<:Real

Reduces a real square matrix `A` to upper Hessenberg form using Householder transformations.

An upper Hessenberg matrix is a square matrix where all entries below the first subdiagonal are zero.
This reduction is typically used as a preprocessing step for efficient QR-based eigenvalue algorithms,
as it preserves the eigenvalues of the original matrix and speeds up convergence.

# Arguments
- `A`: A real square matrix to be reduced.

# Returns
- A matrix `H` that is similar to `A` and is in upper Hessenberg form.

# Errors
- Throws an error if the input matrix is not square.

# Examples
```julia
julia> A = rand(3,3); H = hessenberg(A);

julia> H[3,1] == 0  # All elements below first subdiagonal are zero
true
```
"""
function hessenberg(A::AbstractMatrix{T}) where T<:Real
    if !issquare(A)
        error("Hessenberg reduction requires a square matrix")
    end

    n = size(A, 1)
    H = copy(A)

    for k in 1:n-2
        x = H[k+1:end, k]

        # Skip if subvector already aligned (avoids division by zero)
        if norm(x[2:end]) < eps(T)
            continue
        end

        # Construct Householder vector to zero out below-diagonal entries in column k
        v = copy(x)
        v[1] += sign(x[1]) * norm(x)
        v = v / norm(v)

        # Apply the Householder transformation from the left:
        # Reflect rows k+1 to n of column k through the subspace orthogonal to v
        H[k+1:end, k:end] .-= 2 * v * (v' * H[k+1:end, k:end])

        # Apply the Householder transformation from the right:
        # Ensure similarity transformation: H ← QᵗHQ
        H[:, k+1:end] .-= 2 * (H[:, k+1:end] * v) * v'
    end

    return H
end


function eigen(A::AbstractMatrix{T}) where T<:Real
    if !issquare(A)
        error("A has to be square to have eigenvalues.")
    end
    # TODO: Implement more methods
    return qr_eigenvalues_iterative(A)
end

"""
    qr_eigenvalues_iterative(A::AbstractMatrix{T}; maxiter::Int = 10000, tol::T = 1e-10) where T<:Real

Computes the eigenvalues of a real square matrix `A` using the QR algorithm with Wilkinson shifts.

This function internally reduces `A` to upper Hessenberg form and performs QR iterations with shifts
that accelerate convergence. The algorithm supports detection of complex conjugate eigenvalues through
2×2 blocks in the real Schur form. Wilkinson shifts target eigenvalues near the bottom-right corner
and help deflate the matrix efficiently.

# Arguments
- `A`: A real square matrix.
- `maxiter`: Maximum number of QR iterations (default: 10000).
- `tol`: Tolerance for off-diagonal convergence (default: 1e-10).

# Returns
- A vector of complex eigenvalues (type `Vector{ComplexF64}`), including real and complex conjugate pairs.

# Errors
- Throws an error if the iteration does not converge within `maxiter` steps.

# Example
```julia
julia> A = [4.0 2.0 -1.0; 0.0 3.0 2.0; 0.0 0.0 1.0];
       qr_eigenvalues_iterative(A)
3-element Vector{ComplexF64}:
 4.0 + 0.0im
 3.0 + 0.0im
 1.0 + 0.0im
```
"""
function qr_eigenvalues_iterative(A::AbstractMatrix{T}; maxiter::Int = 10000, tol::T = 1e-10) where T<:Real

    H = hessenberg(A)  # Reduce to upper Hessenberg for efficiency

    # Compute a Wilkinson shift based on the bottom-right 2x2 block
    function wilkinson_shift(H)
        n = size(H, 1)
        a = H[n-1, n-1]
        b = H[n-1, n]
        c = H[n, n-1]
        d = H[n, n]
        tr = a + d
        det = a*d - b*c
        discr = complex(tr^2 - 4det)
        eig1 = (tr + sqrt(discr)) / 2
        eig2 = (tr - sqrt(discr)) / 2
        return abs(eig1 - d) < abs(eig2 - d) ? real(eig1) : real(eig2)
    end

    # Returns a function that switches from unshifted to shifted QR after a few iterations
    function make_qr_stepper(; T, shift_start = 10)
        count = Ref(0)
        function qr_step(H)
            count[] += 1
            if count[] < shift_start
                Q, R = qr(H)
                return R * Q
            else
                shift = wilkinson_shift(H)
                Q, R = qr(H - shift * eye(T, size(H, 1)))
                return R * Q + shift * eye(T, size(H, 1))
            end
        end
        return qr_step
    end

    f = make_qr_stepper(T = T, shift_start = 10)

    # Measures off-diagonal residuals, allowing 2×2 blocks in the structure
    function quasi_upper_triangular_residual(H)
        n = size(H, 1)
        residual = 0.0
        i = 1
        while i < n
            v = abs(H[i+1,i])
            if v < tol                      # Converged to 1x1 block
                i += 1 
            elseif i == n-1                 # Bottom 2x2 block
                i += 1
            elseif abs(H[i+2,i+1]) < tol    # Isolated 2x2 block
                i += 2
            else                            # Not yet converged
                residual += v^2
                i += 1
            end
        end
        return sqrt(residual)
    end

    # Run the QR iteration loop
    H_final, _ = iterative_solver(f, H; tol = tol, max_iter = maxiter, custom_norm = quasi_upper_triangular_residual)

    n = size(H_final, 1)
    eigenvalues = ComplexF64[]
    i = 1

    # Extract eigenvalues from 1x1 or 2x2 blocks
    while i <= n
        if i < n && abs(H_final[i+1, i]) > tol
            # 2x2 block → compute eigenvalues of submatrix
            B = H_final[i:i+1, i:i+1]
            trace = B[1,1] + B[2,2]
            det = B[1,1]*B[2,2] - B[1,2]*B[2,1]
            realpart = trace / 2
            imagpart = sqrt(complex(trace^2 - 4det)) / 2
            push!(eigenvalues, realpart + imagpart, realpart - imagpart)
            i += 2
        else
            push!(eigenvalues, H_final[i, i])  # 1x1 block → real eigenvalue
            i += 1
        end
    end

    return eigenvalues
end



function spectral_radius(A::AbstractMatrix{T}) where T<:Real
    return maximum(abs, eigen(A))
end