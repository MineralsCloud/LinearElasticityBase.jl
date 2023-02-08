using ConstructionBase: constructorof
using StaticArrays: MVector
using Tensorial: SymmetricSecondOrderTensor, SymmetricFourthOrderTensor

export TensorStress,
    TensorStrain,
    StiffnessTensor,
    ComplianceTensor,
    EngineeringStress,
    EngineeringStrain,
    ComplianceMatrix,
    StiffnessMatrix

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
    data::MVector{6,T}
end
EngineeringStress(v::AbstractVector) = EngineeringStress(MVector{6}(v))
EngineeringStress(data...) = EngineeringStress(MVector{6}(data...))
struct EngineeringStrain{T} <: Strain{T,1}
    data::MVector{6,T}
end
EngineeringStrain(v::AbstractVector) = EngineeringStrain(MVector{6}(v))
EngineeringStrain(data...) = EngineeringStrain(MVector{6}(data...))
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

Base.getindex(A::Union{Stress,Strain,Stiffness,Compliance}, i) = getindex(parent(A), i)

Base.setindex!(A::Union{EngineeringStress,EngineeringStrain}, v, i) =
    setindex!(parent(A), v, i)

Base.parent(A::Union{Stress,Strain,Stiffness,Compliance}) = A.data

Base.IndexStyle(::Type{<:Union{Stress,Strain,Stiffness,Compliance}}) = IndexLinear()

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
    @eval begin
        Base.BroadcastStyle(::Type{<:$T}) = Broadcast.ArrayStyle{$T}()
        Base.similar(
            bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{$T}}, ::Type{S}
        ) where {S} = similar($T{S}, axes(bc))
        $T{S}(::UndefInitializer, dims) where {S} = $T(Array{S,length(dims)}(undef, dims))
    end
end

function Base.similar(A::Union{EngineeringStress,EngineeringStrain}, ::Type{S}) where {S}
    T = constructorof(typeof(A))
    return T{S}(Vector{S}(undef, size(A)))
end
function Base.similar(A::Union{TensorStress,TensorStrain}, ::Type{S}) where {S}
    T = constructorof(typeof(A))
    return T{S}(Matrix{S}(undef, size(A)))
end
