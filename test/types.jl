@testset "Promote all values before constructing types (#25)" begin
    e = EngineeringStrain([j == 1 ? 0.005 : 0 for j in 1:6])
    @test eltype(e) == Float64
    e = EngineeringStrain([j == 1 ? 1//2 : 0 for j in 1:6])
    @test eltype(e) == Rational{Int}
end
