export isuniaxial, isbiaxial

# No shear components, only one normal component is nonzero
isuniaxial(x::Union{EngineeringStress,EngineeringStrain}) =
    iszero(x[4:end]) && length(filter(iszero, x[1:3])) == 2
isuniaxial(σ::TensorStress) = isuniaxial(EngineeringStress(σ))
isuniaxial(ε::TensorStrain) = isuniaxial(EngineeringStrain(ε))

function isbiaxial(x::Union{EngineeringStress,EngineeringStrain})
    n = length(filter(!iszero, x[4:end]))
    return if n > 1
        false  # Triaxial
    elseif n == 1
        if all(iszero, x[1:3])  # Pure shear
            true
        else
            if !iszero(x[6])  # 12 ≠ 0
                !iszero(x[1]) && !iszero(x[2]) && iszero(x[3])
            elseif !iszero(x[5])  # 13 ≠ 0
                !iszero(x[1]) && iszero(x[2]) && !iszero(x[3])
            else  # 23 ≠ 0
                iszero(x[1]) && !iszero(x[2]) && !iszero(x[3])
            end
        end
    else  # `n` is zero, no shear components
        length(filter(iszero, x[1:3])) == 1  # Only two normal components
    end
end
isbiaxial(σ::TensorStress) = isbiaxial(EngineeringStress(σ))
isbiaxial(ε::TensorStrain) = isbiaxial(EngineeringStrain(ε))
