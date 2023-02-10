# See https://discourse.julialang.org/t/how-to-compare-two-vectors-whose-elements-are-equal-but-their-types-are-not-the-same/
@testset "Test equality" begin
    @test EngineeringStrain(1:6) == EngineeringStrain(1:6)
    @test EngineeringStrain(1:6) !== EngineeringStrain(1:6)
    @test !(EngineeringStrain(1:6) ⩵ EngineeringStress(1:6))  # Different types
    @test EngineeringStrain(1:6) !== EngineeringStrain(float(1:6))
end

@testset "Test `zero`" begin
    ϵ = EngineeringStrain(1:6)
    @test zero(ϵ) == EngineeringStrain(0, 0, 0, 0, 0, 0)
    @test zero(float(ϵ)) == float(EngineeringStrain(0, 0, 0, 0, 0, 0))
    σ = EngineeringStress(1:6)
    @test zero(σ) == EngineeringStress(0, 0, 0, 0, 0, 0)
    @test zero(float(σ)) == float(EngineeringStress(0, 0, 0, 0, 0, 0))
    σ = TensorStress(EngineeringStress(1:6))
    @test zero(σ) == zeros(Int, 3, 3)
end

@testset "Test negation" begin
    @test typeof(-EngineeringStrain(1:6)) == EngineeringStrain{Int}
    @test typeof(-EngineeringStrain(1.0:6.0)) == EngineeringStrain{Float64}
    @test -EngineeringStrain(1:6) == EngineeringStrain(-1, -2, -3, -4, -5, -6)
    @test -EngineeringStrain(1:6) == EngineeringStrain(-1, -2, -3, -4, -5, -6)
    @test -EngineeringStress(1.0:6.0) == EngineeringStress(-1.0, -2, -3, -4, -5, -6)
    @test -(-EngineeringStress(1.0:6.0)) == EngineeringStress(1.0:6.0)
end

@testset "Test multiplication and division" begin
    ϵ = EngineeringStrain(1:6)
    @test typeof(2ϵ) == EngineeringStrain{Int}
    @test typeof(2.0ϵ) == EngineeringStrain{Float64}
    @test 2ϵ == 2 * ϵ == ϵ * 2 == 2.0ϵ == EngineeringStrain(2:2:12)
    @test ϵ / 2 == EngineeringStrain(0.5, 1, 1.5, 2, 2.5, 3)
    @test 2ϵ / 2 == ϵ
end

@testset "Test addition and subtraction" begin
    @testset "Test 1" begin
        ϵ = EngineeringStrain(1:6)
        ϵ′ = -EngineeringStrain(1:6)
        @test ϵ - ϵ′ == 2ϵ
        @test ϵ′ - ϵ == -2ϵ
        @test ϵ + ϵ′ == zero(ϵ)
    end
    @testset "Test 2" begin
        ϵ = EngineeringStrain(1:6)
        ϵ′ = zero(ϵ)
        @test typeof(ϵ + ϵ) == EngineeringStrain{Int}
        @test typeof(ϵ - ϵ) == EngineeringStrain{Int}
        @test typeof(ϵ + ϵ′) == EngineeringStrain{Int}
        @test typeof(ϵ - ϵ′) == EngineeringStrain{Int}
        @test ϵ + ϵ′ == ϵ′ + ϵ == ϵ
        @test ϵ - ϵ′ == -(ϵ′ - ϵ) == ϵ
    end
    @testset "Test 3" begin
        σ = TensorStress(EngineeringStress(1:6))
        σ′ = -TensorStress(EngineeringStress(1:6))
        @test σ - σ′ == 2σ
        @test σ′ - σ == -2σ
        @test σ + σ′ == TensorStress(zeros(Int, 3, 3))
    end
end
