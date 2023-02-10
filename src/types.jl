using StaticArrays: MVector
using Tensorial: SymmetricSecondOrderTensor, SymmetricFourthOrderTensor

export TensorStress,
    TensorStrain,
    StiffnessTensor,
    ComplianceTensor,
    EngineeringStress,
    EngineeringStrain,
    ComplianceMatrix,
    StiffnessMatrix,
    isequivalent,
    ⩵

abstract type Stress{T,N} <: AbstractArray{T,N} end
abstract type Strain{T,N} <: AbstractArray{T,N} end
abstract type Stiffness{T,N} <: AbstractArray{T,N} end
abstract type Compliance{T,N} <: AbstractArray{T,N} end
struct TensorStress{T} <: Stress{T,2}
    data::SymmetricSecondOrderTensor{3,T,6}
end
TensorStress(data::AbstractMatrix) = TensorStress(SymmetricSecondOrderTensor{3}(data))
TensorStress(values...) = TensorStress(SymmetricSecondOrderTensor{3}(values...))
struct TensorStrain{T} <: Strain{T,2}
    data::SymmetricSecondOrderTensor{3,T,6}
end
TensorStrain(data::AbstractMatrix) = TensorStrain(SymmetricSecondOrderTensor{3}(data))
TensorStrain(values...) = TensorStrain(SymmetricSecondOrderTensor{3}(values...))
struct StiffnessTensor{T} <: Stiffness{T,4}
    data::SymmetricFourthOrderTensor{3,T}
end
struct ComplianceTensor{T} <: Compliance{T,4}
    data::SymmetricFourthOrderTensor{3,T}
end
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
    data::SymmetricSecondOrderTensor{6,T,21}
end
StiffnessMatrix(data::AbstractMatrix) = StiffnessMatrix(SymmetricSecondOrderTensor{6}(data))
StiffnessMatrix(values...) = StiffnessMatrix(SymmetricSecondOrderTensor{6}(values...))
struct ComplianceMatrix{T} <: Compliance{T,2}
    data::SymmetricSecondOrderTensor{6,T,21}
end
ComplianceMatrix(data::AbstractMatrix) =
    ComplianceMatrix(SymmetricSecondOrderTensor{6}(data))
ComplianceMatrix(values...) = ComplianceMatrix(SymmetricSecondOrderTensor{6}(values...))

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

for T in (:EngineeringStress, :EngineeringStrain)
    @eval Base.similar(::$T, ::Type{S}, dims::Dims) where {S} = $T(Vector{S}(undef, dims))
end

# See https://discourse.julialang.org/t/how-to-compare-two-vectors-whose-elements-are-equal-but-their-types-are-not-the-same/94309/6
for (S, T) in (
    (:EngineeringStress, :EngineeringStrain),
    (:TensorStress, :TensorStrain),
    (:StiffnessTensor, :ComplianceTensor),
    (:StiffnessMatrix, :ComplianceMatrix),
)
    @eval begin
        isequivalent(s::$S, t::$T) = false
        isequivalent(t::$T, s::$S) = isequivalent(s, t)
    end
end
isequivalent(x, y) = x == y  # Fallback
const ⩵ = isequivalent
