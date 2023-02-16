using LinearAlgebra: Eigen, eigen
using Tensorial: stress_invariants, deviatoric_stress_invariants, vol, dev

export principal_values,
    principal_axes, principal_invariants, main_invariants, hydrostatic, deviatoric

for T in (:TensorStress, :TensorStrain)
    @eval begin
        principal_values(x::$T) = _eigen(x).values
        principal_axes(x::$T) = _eigen(x).vectors
        principal_invariants(x::$T) = stress_invariants(_tensor(x))
        main_invariants(x::$T) = deviatoric_stress_invariants(_tensor(x))
        hydrostatic(x::$T) = $T(vol(_tensor(x)))
        deviatoric(x::$T) = $T(dev(_tensor(x)))
    end
end

_tensor(x) = SymmetricSecondOrderTensor{3}(float(x))

function _eigen(x)
    x′ = @. x / oneunit(x)  # Dimensionless
    eg = eigen(Matrix(x′))
    return Eigen(eg.values * oneunit(eltype(x)), eg.vectors)
end
