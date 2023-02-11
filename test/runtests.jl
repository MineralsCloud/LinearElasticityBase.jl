using LinearElasticityBase
using Test

@testset "LinearElasticityBase.jl" begin
    include("operations.jl")
    include("invariants.jl")
    include("axial.jl")
end
