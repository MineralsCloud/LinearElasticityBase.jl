using ConstructionBase: constructorof
using Tensorial: SymmetricSecondOrderTensor, SymmetricFourthOrderTensor, Vec

abstract type Stress{T,N} <: AbstractArray{T,N} end
abstract type Strain{T,N} <: AbstractArray{T,N} end
abstract type Stiffness{T,N} <: AbstractArray{T,N} end
abstract type Compliance{T,N} <: AbstractArray{T,N} end
struct TensorStress{T} <: Stress{T,2}
    data::SymmetricSecondOrderTensor{3,T,6}
end
TensorStress(m::AbstractMatrix) = TensorStress(SymmetricSecondOrderTensor{3}(m))
TensorStress(data...) = TensorStress(SymmetricSecondOrderTensor{3}(data...))
struct TensorStrain{T} <: Strain{T,2}
    data::SymmetricSecondOrderTensor{3,T,6}
end
TensorStrain(m::AbstractMatrix) = TensorStrain(SymmetricSecondOrderTensor{3}(m))
TensorStrain(data...) = TensorStrain(SymmetricSecondOrderTensor{3}(data...))
struct StiffnessTensor{T} <: Stiffness{T,4}
    data::SymmetricFourthOrderTensor{3,T}
end
struct ComplianceTensor{T} <: Compliance{T,4}
    data::SymmetricFourthOrderTensor{3,T}
end
struct EngineeringStress{T} <: Stress{T,1}
    data::Vec{6,T}
end
EngineeringStress(v::AbstractVector) = EngineeringStress(Vec{6}(v))
EngineeringStress(data...) = EngineeringStress(Vec{6}(data...))
struct EngineeringStrain{T} <: Strain{T,1}
    data::Vec{6,T}
end
EngineeringStrain(v::AbstractVector) = EngineeringStrain(Vec{6}(v))
EngineeringStrain(data...) = EngineeringStrain(Vec{6}(data...))
struct StiffnessMatrix{T} <: Stiffness{T,2}
    data::SymmetricSecondOrderTensor{6,T,21}
end
StiffnessMatrix(m::AbstractMatrix) = StiffnessMatrix(SymmetricSecondOrderTensor{6}(m))
StiffnessMatrix(data...) = StiffnessMatrix(SymmetricSecondOrderTensor{6}(data...))
struct ComplianceMatrix{T} <: Compliance{T,2}
    data::SymmetricSecondOrderTensor{6,T,21}
end
ComplianceMatrix(m::AbstractMatrix) = ComplianceMatrix(SymmetricSecondOrderTensor{6}(m))
ComplianceMatrix(data...) = ComplianceMatrix(SymmetricSecondOrderTensor{6}(data...))

Base.size(::Union{TensorStress,TensorStrain}) = (3, 3)
Base.size(::Union{StiffnessTensor,ComplianceTensor}) = (3, 3, 3, 3)
Base.size(::Union{EngineeringStress,EngineeringStrain}) = (6,)
Base.size(::Union{StiffnessMatrix,ComplianceMatrix}) = (6, 6)

Base.getindex(A::Union{Stress,Strain,Stiffness,Compliance}, i) = getindex(A.data, i)

Base.setindex!(A::Union{Stress,Strain,Stiffness,Compliance}, v, i) =
    setindex!(parent(A), v, i)

Base.parent(A::Union{Stress,Strain,Stiffness,Compliance}) = A.data

Base.IndexStyle(::Type{<:Union{Stress,Strain,Stiffness,Compliance}}) = IndexLinear()

function Base.similar(A::Union{EngineeringStress,EngineeringStrain}, ::Type{S}) where {S}
    T = constructorof(typeof(A))
    return T(Vector{S}(undef, size(A)))
end
function Base.similar(A::Union{TensorStress,TensorStrain}, ::Type{S}) where {S}
    T = constructorof(typeof(A))
    return T(Matrix{S}(undef, size(A)))
end

# See https://github.com/JuliaLang/julia/blob/cb9acf5/base/arraymath.jl#L19-L26
for op in (:*, :/)
    @eval Base.$op(A::Union{Stress,Strain,Stiffness,Compliance}, B::Number) =
        constructorof(typeof(A))(Base.broadcast_preserving_zero_d($op, A, B))
end
Base.:*(B::Number, A::Union{Stress,Strain,Stiffness,Compliance}) = A * B

Base.:-(A::Union{Stress,Strain,Stiffness,Compliance}) =
    constructorof(typeof(A))(Base.broadcast_preserving_zero_d(-, A))

for op in (:+, :-)
    for T in (
        :TensorStress,
        :TensorStrain,
        :EngineeringStress,
        :EngineeringStrain,
        :StiffnessMatrix,
        :ComplianceMatrix,
        :StiffnessTensor,
        :ComplianceTensor,
    )
        @eval Base.$op(A::$T, B::$T) = $T(Base.broadcast_preserving_zero_d($op, A, B))
    end
end
