using LinearAlgebra: dot

export energydensity

energydensity(σ::EngineeringStress, ϵ::EngineeringStrain) = dot(σ, ϵ) / 2
energydensity(σ::TensorStress, ε::TensorStrain) = double_contraction(σ, ε) / 2
energydensity(ε, σ) = energydensity(ε, σ)
energydensity(c::StiffnessMatrix, ϵ::EngineeringStrain) = dot(ϵ, c, ϵ) / 2
energydensity(s::ComplianceMatrix, σ::EngineeringStress) = dot(σ, s, σ) / 2
energydensity(c::StiffnessTensor, ε::TensorStrain) =
    energydensity(StiffnessMatrix(c), EngineeringStrain(ε))
energydensity(s::ComplianceTensor, σ::TensorStress) =
    energydensity(ComplianceMatrix(s), EngineeringStress(σ))
