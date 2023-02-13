using Unitful: @u_str

@testset "Test broadcasting" begin
    σ = TensorStress([
        300 100 0
        100 100 0
        0 0 0
    ])
    @test σ * 5.0 == 5.0 * σ == TensorStress([
        300 100 0
        100 100 0
        0 0 0
    ] * 5)
    @test σ * u"MPa" == u"MPa" * σ == TensorStress([
        300 100 0
        100 100 0
        0 0 0
    ] * u"MPa")
    @test σ + σ == 2 * σ == σ * 2
    @test σ - σ == 0 * σ == σ * 0
    @test σ .+ 100 == TensorStress([
        400 200 100
        200 200 100
        100 100 100
    ])
end

@testset "Test `similar`" begin
    σ = TensorStress([
        300 100 0
        100 100 0
        0 0 0
    ]) * u"MPa"
    @test typeof(similar(σ)) == TensorStress{typeof(1u"MPa")}
    @test size(similar(σ)) == (3, 3)
    @test typeof(similar(σ, axes(σ))) == TensorStress{typeof(1u"MPa")}
    @test size(similar(σ, axes(σ))) == (3, 3)
    @test_throws DimensionMismatch similar(σ, 4, 4)
    @test typeof(similar(typeof(σ), 3, 3)) == TensorStress{typeof(1u"MPa")}
    @test size(similar(typeof(σ), 3, 3)) == (3, 3)
    @test_throws DimensionMismatch similar(typeof(σ))
    @test_throws DimensionMismatch similar(typeof(σ), 4, 4)
end
