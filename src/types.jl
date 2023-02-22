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
struct TensorStrain{T} <: Strain{T,2}
    data::MMatrix{3,3,T,9}
end
struct StiffnessTensor{T} <: Stiffness{T,4}
    data::SymmetricFourthOrderTensor{3,T}
end
struct ComplianceTensor{T} <: Compliance{T,4}
    data::SymmetricFourthOrderTensor{3,T}
end
struct EngineeringStress{T} <: Stress{T,1}
    data::MVector{6,T}
end
struct EngineeringStrain{T} <: Strain{T,1}
    data::MVector{6,T}
end
struct StiffnessMatrix{T} <: Stiffness{T,2}
    data::MMatrix{6,6,T,36}
end
struct ComplianceMatrix{T} <: Compliance{T,2}
    data::MMatrix{6,6,T,36}
end

# Constructors
for T in (:EngineeringStress, :EngineeringStrain)
    @eval begin
        $T(data::AbstractVector) = $T(MVector{6}(promote(data...)))
        $T(values...) = $T(collect(values))
    end
end
for (T, N) in
    zip((:TensorStress, :TensorStrain, :StiffnessMatrix, :ComplianceMatrix), (3, 3, 6, 6))
    @eval begin
        $T(data::AbstractMatrix{S}) where {S} = $T{S}(MMatrix{$N,$N}(promote(data...)))
        $T(values...) = $T(SymmetricSecondOrderTensor{$N}(values...))
    end
end
for T in (:StiffnessTensor, :ComplianceTensor)
    @eval $T(data::AbstractArray{S,4}) where {S} =
        $T{S}(SymmetricFourthOrderTensor{3}(promote(data...)))
end

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
                return Array{eltype(S),ndims(S)}(undef, dims)
            end
        end
        # Override https://github.com/JuliaLang/julia/blob/618bbc6/base/abstractarray.jl#L806
        function Base.similar(A::$T, ::Type{S}, dims::Dims) where {S}
            if dims == size(A)
                $T(zeros(S, dims))
            else
                return Array{S,ndims(A)}(undef, dims)
            end
        end
    end
end
