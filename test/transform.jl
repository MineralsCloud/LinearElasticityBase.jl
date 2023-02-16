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

