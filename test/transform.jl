@testset "Test example from notes (3.35)" begin
    for τ in (250, 250u"GPa")
        𝟘 = zero(τ)
        σ = TensorStress([
            𝟘 τ 𝟘
            τ 𝟘 𝟘
            𝟘 𝟘 𝟘
        ])
        Q = [
            1 1 0
            -1 1 0
            0 0 √2
        ] / √2
        @test rotate_axes(σ, Q) ≈ TensorStress([
            τ 𝟘 𝟘
            𝟘 -τ 𝟘
            𝟘 𝟘 𝟘
        ])
        @test rotate_axes(EngineeringStress(σ), Q) ≈ EngineeringStress(τ, -τ, 0, 0, 0, 0)
    end
end

@testset "Test homework 3 question 1" begin
    σ = TensorStress([
        0 0 0
        0 245 0
        0 0 0
    ])
    Q = [
        1 1 0
        -1 1 0
        0 0 √2
    ] / √2
    @test rotate_axes(σ, Q)[1, 2] ≈ sin(2 * pi / 4) / 2 * σ[2, 2]
end

@testset "Test homework 3 question 2" begin
    σ = TensorStress([
        300 100 0
        100 100 0
        0 0 0
    ])
    Q = θ -> [
        cos(θ) sin(θ) 0
        -sin(θ) cos(θ) 0
        0 0 1
    ]
    @testset "Determine the principal stresses" begin
        @test rotate_axes(σ, Q(pi / 8)) ≈ TensorStress([
            200+100 * √2 0 0
            0 200-100 * √2 0
            0 0 0
        ])
    end
    @testset "Determine the maximum in-plane shear" begin
        @test rotate_axes(σ, Q(-pi / 8))[1, 2] == 100 * √2
        @test rotate_axes(σ, Q(3pi / 8))[1, 2] == -100 * √2
    end
end

@testset "Test question 5 in midterm 1, 2019" begin
    σ₀ = 250
    σ = TensorStress([
        σ₀ 0 0
        0 2σ₀ 0
        0 0 0
    ])
    Q = [
        1/√2 -1/√2 0
        1/√6 1/√6 -2/√6
        1/√3 1/√3 1/√3
    ]
    @test rotate_axes(σ, Q)[3, 1] ≈ -σ₀ / √6
end

@testset "Test question 6 in midterm 1, 2019" begin
    σ₀ = 360
    σ = TensorStress([
        σ₀ σ₀/3 0
        σ₀/3 σ₀ 0
        0 0 0
    ])
    Q = θ -> [
        cos(θ) sin(θ) 0
        -sin(θ) cos(θ) 0
        0 0 1
    ]
    @test rotate_axes(σ, Q(-pi / 4)) ≈ TensorStress(diagm([2σ₀ / 3, 4σ₀ / 3, 0]))
    @test rotate_axes(σ, Q(pi / 4)) == TensorStress(diagm([4σ₀ / 3, 2σ₀ / 3, 0]))
    @test principal_values(σ) ≈ [0, 2σ₀ / 3, 4σ₀ / 3]
end

@testset "Test using principal axes as rotation matrix" begin
    ε = TensorStrain([
        0.5 0.3 0.2
        0.3 -0.2 -0.1
        0.2 -0.1 0.1
    ])
    evecs = principal_axes(ε)'
    normal_strains = rotate_axes(ε, evecs)
    @test norm(normal_strains - [
        -0.370577 0 0
        0 0.115308 0
        0 0 0.655269
    ]) < 1e-6
    @test principal_values(ε) ≈ diag(normal_strains)
end
