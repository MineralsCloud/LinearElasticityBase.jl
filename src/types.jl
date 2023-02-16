using StaticArrays: MVector, MMatrix
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
    data::MMatrix{3,3,T,9}
end
TensorStress(data::AbstractMatrix{T}) where {T} = TensorStress{T}(MMatrix{3,3}(data))
TensorStress(values...) = TensorStress(SymmetricSecondOrderTensor{3}(values...))
struct TensorStrain{T} <: Strain{T,2}
    data::MMatrix{3,3,T,9}
end
TensorStrain(data::AbstractMatrix{T}) where {T} = TensorStrain{T}(MMatrix{3,3}(data))
TensorStrain(values...) = TensorStrain(SymmetricSecondOrderTensor{3}(values...))
struct StiffnessTensor{T} <: Stiffness{T,4}
    data::SymmetricFourthOrderTensor{3,T}
end
StiffnessTensor(data::AbstractArray{T,4}) where {T} =
    StiffnessTensor{T}(SymmetricFourthOrderTensor{3}(data))
struct ComplianceTensor{T} <: Compliance{T,4}
    data::SymmetricFourthOrderTensor{3,T}
end
ComplianceTensor(data::AbstractArray{T,4}) where {T} =
    ComplianceTensor{T}(SymmetricFourthOrderTensor{3}(data))
struct EngineeringStress{T} <: Stress{T,1}
    data::MVector{6,T}
end
EngineeringStress(data::AbstractVector) = EngineeringStress(MVector{6}(data))
EngineeringStress(values...) = EngineeringStress(MVector{6}(values...))
struct EngineeringStrain{T} <: Strain{T,1}
    data::MVector{6,T}
end
EngineeringStrain(data::AbstractVector) = EngineeringStrain(MVector{6}(data))
EngineeringStrain(values...) = EngineeringStrain(MVector{6}(values...))
struct StiffnessMatrix{T} <: Stiffness{T,2}
    data::MMatrix{6,6,T,36}
end
StiffnessMatrix(data::AbstractMatrix{T}) where {T} = StiffnessMatrix{T}(MMatrix{6,6}(data))
StiffnessMatrix(values...) = StiffnessMatrix(SymmetricSecondOrderTensor{6}(values...))
struct ComplianceMatrix{T} <: Compliance{T,2}
    data::MMatrix{6,6,T,36}
end
ComplianceMatrix(data::AbstractMatrix{T}) where {T} =
    ComplianceMatrix{T}(MMatrix{6,6}(data))
ComplianceMatrix(values...) = ComplianceMatrix(SymmetricSecondOrderTensor{6}(values...))

Base.size(::Union{TensorStress,TensorStrain}) = (3, 3)
Base.size(::Union{StiffnessTensor,ComplianceTensor}) = (3, 3, 3, 3)
Base.size(::Union{EngineeringStress,EngineeringStrain}) = (6,)
Base.size(::Union{StiffnessMatrix,ComplianceMatrix}) = (6, 6)

Base.getindex(A::Union{Stress,Strain,Stiffness,Compliance}, i) = getindex(parent(A), i)

Base.setindex!(A::Union{EngineeringStress,EngineeringStrain}, v, i) =
    setindex!(parent(A), v, i)
Base.setindex!(
    A::Union{TensorStress,TensorStrain,StiffnessMatrix,ComplianceMatrix}, v, i::Integer
) = setindex!(parent(A), v, i)
# See https://github.com/JuliaLang/LinearAlgebra.jl/blob/13df5a2/src/symmetric.jl#L226-L229
function Base.setindex!(
    A::Union{TensorStress,TensorStrain,StiffnessMatrix,ComplianceMatrix},
    v,
    i::Integer,
    j::Integer,
)
    if i == j
        setindex!(parent(A), v, i, i)
    else
        setindex!(parent(A), v, i, j)
        setindex!(parent(A), v, j, i)
    end
end

Base.parent(A::Union{Stress,Strain,Stiffness,Compliance}) = A.data

Base.IndexStyle(::Type{<:Union{Stress,Strain,Stiffness,Compliance}}) = IndexLinear()

for T in (
    :EngineeringStress,
    :EngineeringStrain,
    :TensorStress,
    :TensorStrain,
    :StiffnessMatrix,
    :ComplianceMatrix,
)
    @eval begin
        Base.BroadcastStyle(::Type{<:$T}) = Broadcast.ArrayStyle{$T}()
        Base.similar(
            bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{$T}}, ::Type{S}
        ) where {S} = similar($T{S}, axes(bc))
        # Override https://github.com/JuliaLang/julia/blob/618bbc6/base/abstractarray.jl#L841
        Base.similar(::Type{S}, dims::Dims) where {S<:$T} = $T(zeros(eltype(S), dims))
        # Override https://github.com/JuliaLang/julia/blob/618bbc6/base/abstractarray.jl#L806
        Base.similar(::$T, ::Type{S}, dims::Dims) where {S} = $T(zeros(S, dims))
    end
end
