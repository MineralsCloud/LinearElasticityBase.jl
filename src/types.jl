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

abstract type Variable{T,N} <: AbstractArray{T,N} end
abstract type ElasticConstants{T,N} <: AbstractArray{T,N} end
abstract type Stress{T,N} <: Variable{T,N} end
abstract type Strain{T,N} <: Variable{T,N} end
abstract type Stiffness{T,N} <: ElasticConstants{T,N} end
abstract type Compliance{T,N} <: ElasticConstants{T,N} end
# See https://github.com/JuliaLang/julia/blob/237c92d/base/array.jl#L38
const TensorVariable{T} = Variable{T,2}
const EngineeringVariable{T} = Variable{T,1}
const ElasticConstantsTensor{T} = ElasticConstants{T,4}
const ElasticConstantsMatrix{T} = ElasticConstants{T,2}
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

Base.size(::Type{<:TensorVariable}) = (3, 3)
Base.size(::Type{<:ElasticConstantsTensor}) = (3, 3, 3, 3)
Base.size(::Type{<:EngineeringVariable}) = (6,)
Base.size(::Type{<:ElasticConstantsMatrix}) = (6, 6)
Base.size(A::TensorVariable) = size(typeof(A))
Base.size(A::ElasticConstantsTensor) = size(typeof(A))
Base.size(A::EngineeringVariable) = size(typeof(A))
Base.size(A::ElasticConstantsMatrix) = size(typeof(A))

Base.getindex(A::Union{Variable,ElasticConstants}, i) = getindex(parent(A), i)

Base.setindex!(A::EngineeringVariable, v, i) = setindex!(parent(A), v, i)
Base.setindex!(A::Union{TensorVariable,ElasticConstantsMatrix}, v, i::Integer) =
    setindex!(parent(A), v, i)
# See https://github.com/JuliaLang/LinearAlgebra.jl/blob/13df5a2/src/symmetric.jl#L226-L229
function Base.setindex!(
    A::Union{TensorVariable,ElasticConstantsMatrix}, v, i::Integer, j::Integer
)
    if i == j
        setindex!(parent(A), v, i, i)
    else
        setindex!(parent(A), v, i, j)
        setindex!(parent(A), v, j, i)
    end
end

Base.parent(A::Union{Variable,ElasticConstants}) = A.data

Base.IndexStyle(::Type{<:Union{Variable,ElasticConstants}}) = IndexLinear()

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
        function Base.similar(::Type{S}, dims::Dims) where {S<:$T}
            if dims == size(S)
                $T(zeros(eltype(S), dims))
            else
                throw(ArgumentError(string("invalid size ", dims, " for type ", typeof(S))))
            end
        end
        # Override https://github.com/JuliaLang/julia/blob/618bbc6/base/abstractarray.jl#L806
        function Base.similar(A::$T, ::Type{S}, dims::Dims) where {S}
            if dims == size(A)
                $T(zeros(eltype(S), dims))
            else
                throw(ArgumentError(string("invalid size ", dims, " for type ", typeof(A))))
            end
        end
    end
end
