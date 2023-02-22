using LinearElasticityBase
using Test

@testset "LinearElasticityBase.jl" begin
    include("types.jl")
    include("similar.jl")
    include("operations.jl")
    include("invariants.jl")
    include("axial.jl")
    include("transform.jl")
end
