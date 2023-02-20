using Tensorial: fromvoigt, ⊡, ⋅

import Tensorial: contraction, double_contraction

export to_tensor, to_voigt, ⩵, ⩶

TensorStress(σ::EngineeringStress) = TensorStress(σ[1], σ[6], σ[5], σ[2], σ[4], σ[3])

EngineeringStress(σ::TensorStress) =
    EngineeringStress(σ[1, 1], σ[2, 2], σ[3, 3], σ[2, 3], σ[1, 3], σ[1, 2])

TensorStrain(ϵ::EngineeringStrain) =
    TensorStrain(ϵ[1], ϵ[6] / 2, ϵ[5] / 2, ϵ[2], ϵ[4] / 2, ϵ[3])

EngineeringStrain(ε::TensorStrain) =
    EngineeringStrain(ε[1, 1], ε[2, 2], ε[3, 3], 2ε[2, 3], 2ε[1, 3], 2ε[1, 2])

StiffnessMatrix(c::StiffnessTensor) = StiffnessMatrix(
    c[1, 1, 1, 1],
    c[1, 1, 2, 2],
    c[1, 1, 3, 3],
    c[1, 1, 2, 3],
    c[1, 1, 1, 3],
    c[1, 1, 1, 2],
    c[2, 2, 2, 2],
    c[2, 2, 3, 3],
    c[2, 2, 2, 3],
    c[2, 2, 1, 3],
    c[2, 2, 1, 2],
    c[3, 3, 3, 3],
    c[3, 3, 2, 3],
    c[3, 3, 1, 3],
    c[3, 3, 1, 2],
    c[2, 3, 2, 3],
    c[2, 3, 1, 3],
    c[2, 3, 1, 2],
    c[1, 3, 1, 3],
    c[1, 3, 1, 2],
    c[1, 2, 1, 2],
)

StiffnessTensor(c::StiffnessMatrix) =
    StiffnessTensor(fromvoigt(SymmetricFourthOrderTensor{3}, c.data))

ComplianceMatrix(s::ComplianceTensor) = ComplianceMatrix(
    s[1, 1, 1, 1],
    s[1, 1, 2, 2],
    s[1, 1, 3, 3],
    2s[1, 1, 2, 3],
    2s[1, 1, 1, 3],
    2s[1, 1, 1, 2],
    s[2, 2, 2, 2],
    s[2, 2, 3, 3],
    2s[2, 2, 2, 3],
    2s[2, 2, 1, 3],
    2s[2, 2, 1, 2],
    s[3, 3, 3, 3],
    2s[3, 3, 2, 3],
    2s[3, 3, 1, 3],
    2s[3, 3, 1, 2],
    4s[2, 3, 2, 3],
    4s[2, 3, 1, 3],
    4s[2, 3, 1, 2],
    4s[1, 3, 1, 3],
    4s[1, 3, 1, 2],
    4s[1, 2, 1, 2],
)

ComplianceTensor(s::ComplianceMatrix) = ComplianceTensor(
    SymmetricFourthOrderTensor{3}(function (i, j, k, l)
        if i == j && k == l
            return s[i, k]
        elseif i != j && k != l  # 4 = 9 - (2+3), 5 = 9 - (1+3), 6 = 9 - (1+2)
            return s[9 - (i + j), 9 - (k + l)] / 4
        elseif i == j && k != l
            return s[i, 9 - (k + l)] / 2
        else  # i != j && k == l
            return s[9 - (i + j), k] / 2
        end
    end)
)

to_tensor(ϵ::EngineeringStrain) = TensorStrain(ϵ)
to_tensor(σ::EngineeringStress) = TensorStress(σ)
to_tensor(c::StiffnessMatrix) = StiffnessTensor(c)
to_tensor(s::ComplianceMatrix) = ComplianceTensor(s)

to_voigt(ε::TensorStrain) = EngineeringStrain(ε)
to_voigt(σ::TensorStress) = EngineeringStress(σ)
to_voigt(c::StiffnessTensor) = StiffnessMatrix(c)
to_voigt(s::ComplianceTensor) = ComplianceMatrix(s)

Base.:*(c::StiffnessMatrix, ϵ::EngineeringStrain) = EngineeringStress(c.data ⋅ ϵ.data)
Base.:*(s::ComplianceMatrix, σ::EngineeringStress) = EngineeringStrain(s.data ⋅ σ.data)

Base.inv(c::StiffnessTensor) = ComplianceTensor(inv(c.data))
Base.inv(s::ComplianceTensor) = StiffnessTensor(inv(s.data))
Base.inv(c::StiffnessMatrix) = ComplianceMatrix(inv(c.data))
Base.inv(s::ComplianceMatrix) = StiffnessMatrix(inv(s.data))

contraction(c::StiffnessTensor, ε::TensorStrain, ::Val{2}) = TensorStress(c.data ⊡ ε.data)
contraction(s::ComplianceTensor, σ::TensorStress, ::Val{2}) = TensorStrain(s.data ⊡ σ.data)

@inline double_contraction(c::StiffnessTensor, ε::TensorStrain) = contraction(c, ε, Val(2))
@inline double_contraction(s::ComplianceTensor, σ::TensorStress) = contraction(s, σ, Val(2))

# See https://discourse.julialang.org/t/how-to-compare-two-vectors-whose-elements-are-equal-but-their-types-are-not-the-same/94309/6
for T in (
    :EngineeringStrain,
    :EngineeringStress,
    :TensorStrain,
    :TensorStress,
    :StiffnessMatrix,
    :StiffnessTensor,
    :ComplianceMatrix,
    :ComplianceTensor,
)
    @eval begin
        ⩵(s::$T, t::$T) = s == t
        ⩵(s, t::$T) = s isa $T && s == t
        ⩵(s::$T, t) = t ⩵ s
        ⩶(s, t::$T) = s === t
        ⩶(s::$T, t) = t ⩶ s
    end
end
for (S, T) in (
    (:EngineeringStress, :TensorStress),
    (:EngineeringStrain, :TensorStrain),
    (:StiffnessMatrix, :StiffnessTensor),
    (:ComplianceMatrix, :ComplianceTensor),
)
    @eval begin
        ⩵(s::$S, t::$T) = to_voigt(t) ⩵ s
        ⩵(t::$T, s::$S) = s ⩵ t
        ⩶(s::$S, t::$T) = to_voigt(t) ⩶ s
        ⩶(t::$T, s::$S) = s ⩶ t
    end
end
