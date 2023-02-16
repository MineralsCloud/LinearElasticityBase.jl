using Einsum: @einsum
using LinearAlgebra: I

export rotate, rotate_basis

for S in (:StiffnessTensor, :ComplianceTensor)
    @eval function rotate(T′::$S, Q::AbstractMatrix)
        @assert isdcm(Q)
        @einsum T[i, j, k, l] := T′[m, n, o, p] * Q[m, i] * Q[n, j] * Q[o, k] * Q[p, l]
        return $S(T)
    end
end
for S in (:TensorStrain, :TensorStress)
    @eval function rotate(T′::$S, Q::AbstractMatrix)
        @assert isdcm(Q)
        @einsum T[i, j] := T′[k, l] * Q[k, i] * Q[l, j]
        return $S(T)
    end
end
for S in (:StiffnessMatrix, :ComplianceMatrix, :EngineeringStrain, :EngineeringStress)
    @eval function rotate(T′::$S, Q::AbstractMatrix)
        t′ = to_tensor(T′)
        T = rotate(t′, Q)
        return to_voigt(T)
    end
end

for S in (:StiffnessTensor, :ComplianceTensor)
    @eval function rotate_basis(T::$S, Q::AbstractMatrix)
        @assert isdcm(Q)
        @einsum T′[i, j, k, l] := T[m, n, o, p] * Q[i, m] * Q[j, n] * Q[k, o] * Q[l, p]
        return $S(T)
    end
end
for S in (:TensorStrain, :TensorStress)
    @eval function rotate_basis(T::$S, Q::AbstractMatrix)
        @assert isdcm(Q)
        @einsum T′[i, j] := T[k, l] * Q[i, k] * Q[j, l]
        return $S(T)
    end
end
for S in (:StiffnessMatrix, :ComplianceMatrix, :EngineeringStrain, :EngineeringStress)
    @eval function rotate_basis(T::$S, Q::AbstractMatrix)
        t = to_tensor(T)
        T′ = rotate_basis(t, Q)
        return to_voigt(T′)
    end
end

"""
    isorthonormal(Q::AbstractMatrix, rtol=√eps)

Test whether `Q` is an orthonormal matrix, with relative tolerance `rtol`.
"""
function isorthonormal(Q::AbstractMatrix, rtol=√eps)
    if Q' * Q == Q * Q' == I
        return true
    else
        return isapprox(Q' * Q, I; rtol=rtol) && isapprox(Q * Q', I; rtol=rtol)
    end
end

"""
    isdcm(Q::AbstractMatrix)

Test whether `Q` is a direction cosine matrix.

A direction cosine matrix is a ``3 \\times 3`` matrix that represent a coordinate
transformation between two orthonormal reference frames. Let those frames be right-handed,
then this transformation is always a rotation.
"""
isdcm(Q::AbstractMatrix) = size(Q) == (3, 3) && isorthonormal(Q)
