
using Test
using Numerik2

@testset "Norm (Vector and Scalar)" begin
    v = [1, 4, -8]
    @test norm(v) ≈ 9.0          # Euclidean
    @test norm(v, 1) == 13.0     # L1
    @test norm(v, Inf) == 8.0
    @test norm(v, -Inf) == 1.0
    @test norm(v, 0) == 3.0
    @test norm(v, -2) ≈ 0.96309 atol=1e-4

    @test norm(3.0) == 3.0       # Scalar fallback
    @test norm(-2.0, 0) == 1.0   # Scalar nonzero count
end

@testset "Fixed Point" begin
    f(x) = cos(x)
    x, n = fixed_point_iteration(f, 0.0, 1e-8)
    @test isapprox(x, f(x), atol = 1e-8)
end

@testset "LU Solve" begin
    A = [2.0 1.0; 4.0 -6.0]
    b = [5.0, -2.0]
    x = linear_solve(A, b; method = "lu")
    @test A * x ≈ b
end

@testset "QR Overdetermined" begin
    A = [1.0 1.0; 1.0 -1.0; 2.0 0.0]
    b = [2.0, 0.0, 4.0]
    x = linear_solve(A, b; method = "qr_over")
    @test norm(A * x - b) ≈ 1.1547 atol=1e-4
end

@testset "QR Underdetermined" begin
    A = [1.0 2.0 3.0; 4.0 5.0 6.0]
    b = [1.0, 0.0]
    x = linear_solve(A, b; method = "qr_under")
    @test isapprox(A * x, b, atol = 1e-12)
    # TODO: minimal norm test
end

@testset "Cholesky SPD Solve" begin
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    x = linear_solve(A, b; method = "spd")
    @test A * x ≈ b
end

@testset "Triangular Systems" begin
    U = [2.0 1.0; 0.0 3.0]
    L = [2.0 0.0; -1.0 3.0]
    x₁ = linear_solve(U, [5.0, 6.0]; method = "triu")
    x₂ = linear_solve(L, [4.0, 5.0]; method = "tril")
    @test U * x₁ ≈ [5.0, 6.0]
    @test L * x₂ ≈ [4.0, 5.0]
end
