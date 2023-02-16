using Einsum: @einsum

export rotate, rotate_basis

for S in (:StiffnessTensor, :ComplianceTensor)
    @eval function rotate(T′::$S, a::AbstractMatrix)
        @assert size(a) == (3, 3)
        @einsum T[i, j, k, l] := T′[m, n, o, p] * a[m, i] * a[n, j] * a[o, k] * a[p, l]
        return $S(T)
    end
end
for S in (:TensorStrain, :TensorStress)
    @eval function rotate(T′::$S, a::AbstractMatrix)
        @assert size(a) == (3, 3)
        @einsum T[i, j] := T′[k, l] * a[k, i] * a[l, j]
        return $S(T)
    end
end
function rotate(T′::Union{EngineeringVariable,ElasticConstantsMatrix}, a::AbstractMatrix)
    t′ = to_tensor(T′)
    T = rotate(t′, a)
    return to_voigt(T)
end

for S in (:StiffnessTensor, :ComplianceTensor)
    @eval function rotate_basis(T::$S, a::AbstractMatrix)
        @assert size(a) == (3, 3)
        @einsum T′[i, j, k, l] := T[m, n, o, p] * a[i, m] * a[j, n] * a[k, o] * a[l, p]
        return $S(T)
    end
end
for S in (:TensorStrain, :TensorStress)
    @eval function rotate_basis(T::$S, a::AbstractMatrix)
        @assert size(a) == (3, 3)
        @einsum T′[i, j] := T[k, l] * a[i, k] * a[j, l]
        return $S(T)
    end
end
function rotate_basis(
    T::Union{EngineeringVariable,ElasticConstantsMatrix}, a::AbstractMatrix
)
    t = to_tensor(T)
    T′ = rotate_basis(t, a)
    return to_voigt(T′)
end

"""
    isorthonormal(Q::AbstractMatrix)

Test whether `Q` is an orthonormal matrix.
"""
isorthonormal(Q::AbstractMatrix) = Q' * Q == Q * Q' == I

"""
    isdcm(Q::AbstractMatrix)

Test whether `Q` is a direction cosine matrix.

A direction cosine matrix is a ``3 \\times 3`` matrix that represent a coordinate
transformation between two orthonormal reference frames. Let those frames be right-handed,
then this transformation is always a rotation.
"""
isdcm(Q::AbstractMatrix) = size(Q) == (3, 3) && isorthonormal(Q)
