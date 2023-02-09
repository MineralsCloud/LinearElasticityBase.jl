# See https://discourse.julialang.org/t/how-to-compare-two-vectors-whose-elements-are-equal-but-their-types-are-not-the-same/
@testset "Test equality" begin
    @test EngineeringStrain(1:6) == EngineeringStrain(1:6)
    @test EngineeringStrain(1:6) !== EngineeringStrain(1:6)
    @test EngineeringStrain(1:6) != EngineeringStress(1:6)  # Different types
    @test EngineeringStrain(1:6) !== EngineeringStrain(float(1:6))
    @test !isequal(EngineeringStrain(1:6), EngineeringStress(1:6))  # Different types
    @test EngineeringStrain(1:6) ≉ EngineeringStress(1:6)  # Different types
end
