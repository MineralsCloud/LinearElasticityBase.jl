using Einsum: @einsum
using LinearAlgebra: I

export rotate, rotate_axes

for S in (:StiffnessTensor, :ComplianceTensor)
    @eval function rotate(T′::$S, Q::AbstractMatrix)
        @assert isrotation(Q)
        @einsum T[i, j, k, l] := T′[m, n, o, p] * Q[m, i] * Q[n, j] * Q[o, k] * Q[p, l]
        return $S(T)
    end
end
for S in (:TensorStrain, :TensorStress)
    @eval function rotate(T′::$S, Q::AbstractMatrix)
        @assert isrotation(Q)
        @einsum T[i, j] := T′[k, l] * Q[k, i] * Q[l, j]
        return $S(T)
    end
end
for S in (:StiffnessMatrix, :ComplianceMatrix, :EngineeringStrain, :EngineeringStress)
    @eval function rotate(x::$S, Q::AbstractMatrix)
        T′ = to_tensor(x)
        T = rotate(T′, Q)
        return to_voigt(T)
    end
end

for S in (:StiffnessTensor, :ComplianceTensor)
    @eval function rotate_axes(T::$S, Q::AbstractMatrix)
        @assert isrotation(Q)
        @einsum T′[i, j, k, l] := T[m, n, o, p] * Q[i, m] * Q[j, n] * Q[k, o] * Q[l, p]
        return $S(T′)
    end
end
for S in (:TensorStrain, :TensorStress)
    @eval function rotate_axes(T::$S, Q::AbstractMatrix)
        @assert isrotation(Q)
        @einsum T′[i, j] := T[k, l] * Q[i, k] * Q[j, l]
        return $S(T′)
    end
end
for S in (:StiffnessMatrix, :ComplianceMatrix, :EngineeringStrain, :EngineeringStress)
    @eval function rotate_axes(x::$S, Q::AbstractMatrix)
        T = to_tensor(x)
        T′ = rotate_axes(T, Q)
        return to_voigt(T′)
    end
end

"""
    isorthonormal(Q::AbstractMatrix)

Test whether `Q` is an orthonormal matrix.
"""
function isorthonormal(Q::AbstractMatrix)
    rtol = get_rtol()
    if Q' * Q == Q * Q' == I
        return true
    else
        return isapprox(Q' * Q, I; rtol=rtol) && isapprox(Q * Q', I; rtol=rtol)
    end
end

"""
    isrotation(Q::AbstractMatrix)

Test whether `Q` is a rotation matrix.

A direction cosine matrix is a ``3 \\times 3`` matrix that represent a coordinate
transformation between two orthonormal reference frames. Let those frames be right-handed,
then this transformation is always a rotation.
"""
isrotation(Q::AbstractMatrix) = size(Q) == (3, 3) && isorthonormal(Q)

# See https://github.com/KristofferC/OhMyREPL.jl/blob/f682498/src/BracketInserter.jl#L44-L45
const RTOL = Ref(sqrt(eps()))

get_rtol() = RTOL[]

set_rtol(v::Real) = RTOL[] = v
