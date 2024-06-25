module QuantumUtils

export GATES
export get_gate
export ⊗
export vec⁻¹
export operators_from_dict
export kron_from_dict
export apply
export haar_random
export haar_identity
export qubit_system_state
export lift
export ket_to_iso
export iso_to_ket
export operator_to_iso_vec
export iso_vec_to_operator
export iso_vec_to_iso_operator
export iso_operator_to_iso_vec
export annihilate
export create
export quad
export cavity_state
export multimode_state
export number
export fidelity
export iso_fidelity
export unitary_fidelity
export population
export populations
export quantum_state

using TrajectoryIndexingUtils
using LinearAlgebra
using SparseArrays


@doc raw"""
    ⊗(A::AbstractVecOrMat, B::AbstractVecOrMat) = kron(A, B)

The Kronecker product, denoted by `⊗`, results in a block matrix formed by multiplying each element of `A` by the entire matrix `B`.

```julia
julia> GATES[:X] ⊗ GATES[:Y]
4×4 Matrix{ComplexF64}:
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0-1.0im
 0.0+0.0im  0.0+0.0im  0.0+1.0im  0.0+0.0im
 0.0+0.0im  0.0-1.0im  0.0+0.0im  0.0+0.0im
 0.0+1.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
```
"""
⊗(A::AbstractVecOrMat, B::AbstractVecOrMat) = kron(A, B)


@doc raw"""
A constant dictionary `GATES` containing common quantum gate matrices as complex-valued matrices.

- `GATES[:I]` - Identity gate: Leaves the state unchanged.
- `GATES[:X]` - Pauli-X (NOT) gate: Flips the qubit state.
- `GATES[:Y]` - Pauli-Y gate: Rotates the qubit state around the Y-axis of the Bloch sphere.
- `GATES[:Z]` - Pauli-Z gate: Flips the phase of the qubit state.
- `GATES[:H]` - Hadamard gate: Creates superposition by transforming basis states.
- `GATES[:CX]` - Controlled-X (CNOT) gate: Flips the second qubit (target) if the first qubit (control) is |1⟩.
- `GATES[:XI]` - Complex gate: A specific gate used for complex operations.
- `GATES[:sqrtiSWAP]` - Square root of iSWAP gate: Partially swaps two qubits with a phase.

Each gate is represented by its unitary matrix.

```julia
julia> GATES[:X]
2×2 Matrix{ComplexF64}:
 0.0+0.0im  1.0+0.0im
 1.0+0.0im  0.0+0.0im

julia> GATES[:Y]
2×2 Matrix{ComplexF64}:
 0.0+0.0im  0.0-1.0im
 0.0+1.0im  0.0+0.0im

julia> GATES[:Z]
2×2 Matrix{ComplexF64}:
 1.0+0.0im   0.0+0.0im
 0.0+0.0im  -1.0+0.0im

julia> get_gate(:CX)
4×4 Matrix{ComplexF64}:
 1.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  1.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  1.0+0.0im
 0.0+0.0im  0.0+0.0im  1.0+0.0im  0.0+0.0im
```
"""
const GATES = Dict{Symbol, Matrix{ComplexF64}}(
    :I => [1 0;
           0 1],

    :X => [0 1;
           1 0],

    :Y => [0 -im;
           im 0],

    :Z => [1 0;
           0 -1],

    :H => [1 1;
           1 -1]/√2,

    :CX => [1 0 0 0;
            0 1 0 0;
            0 0 0 1;
            0 0 1 0],

    :XI => [0 0 -im 0;
            0 0 0 -im;
            -im 0 0 0;
            0 -im 0 0],

    :sqrtiSWAP => [1 0 0 0;
                   0 1/sqrt(2) 1im/sqrt(2) 0;
                   0 1im/sqrt(2) 1/sqrt(2) 0;
                   0 0 0 1]
)

get_gate(U::Symbol) = GATES[U]

@doc raw"""
    apply(gate::Symbol, ψ::Vector{<:Number})

Apply a quantum gate `gate` to a state vector `ψ`.
"""
function apply(gate::Symbol, ψ::Vector{<:Number})
    @assert norm(ψ) ≈ 1.0
    @assert gate in keys(GATES) "gate not found"
    Û = get_gate(gate)
    @assert size(Û, 2) == size(ψ, 1) "gate size does not match ket dim"
    return ComplexF64.(normalize(Û * ψ))
end

@doc raw"""
    haar_random(n::Int)

Generate a random unitary matrix using the Haar measure for an `n`-dimensional system.
"""
function haar_random(n::Int)
    # Ginibre matrix
    Z = (randn(n, n) + im * randn(n, n)) / √2
    F = qr(Z)
    # QR correction (R main diagonal is real, strictly positive)
    Λ = diagm(diag(F.R) ./ abs.(diag(F.R)))
    return F.Q * Λ
end

@doc raw"""
    haar_identity(n::Int, radius::Number)

Generate a random unitary matrix close to the identity matrix using the Haar measure for an `n`-dimensional system with a given `radius`.
"""
function haar_identity(n::Int, radius::Number)
    # Ginibre matrix
    Z = (I + radius * (randn(n, n) + im * randn(n, n)) / √2) / (1 + radius)
    F = qr(Z)
    # QR correction (R main diagonal is real, strictly positive)
    Λ = diagm(diag(F.R) ./ abs.(diag(F.R)))
    return F.Q * Λ
end

@doc raw"""
operators_from_dict(keys::AbstractVector{<:Any}, operator_dictionary; I_key=:I)

    Replace the vector of keys using the operators from a dictionary.
"""
function operators_from_dict(keys::AbstractVector{<:Any}, operator_dictionary; I_key=:I)
    first_operator = first(values(operator_dictionary))
    I_default = Matrix{eltype(first_operator)}(I, size(first_operator))
    # Identity key is replaced by operator_dictionary, else default I.
    return replace(replace(keys, operator_dictionary...), I_key => I_default)
end

@doc raw"""
operators_from_dict(key_string::String, operator_dictionary; I_key="I")

    Replace the string (each character is one key) with operators from a dictionary.
"""
operators_from_dict(key_string::String, operator_dictionary; I_key="I") =
    operators_from_dict([string(c) for c ∈ key_string], operator_dictionary, I_key=I_key)

@doc raw"""
kron_from_dict(keys, dict; kwargs...)

    Reduce the keys to a single operator by using the provided dictionary and the kronecker product.
"""
function kron_from_dict(keys, dict; kwargs...)
    if occursin("+", keys)
        return sum(
            [kron_from_dict(string(s), dict; kwargs...)
            for s ∈ split(keys, "+")]
        )
    else
        return reduce(kron, operators_from_dict(keys, dict; kwargs...))
    end
end

@doc raw"""
    qubit_system_state(ket::String)

Get the state vector for a qubit system given a ket string `ket` of 0s and 1s.
"""
function qubit_system_state(ket::String)
    cs = [c for c ∈ ket]
    @assert all(c ∈ "01" for c ∈ cs)
    states = [c == '0' ? [1, 0] : [0, 1] for c ∈ cs]
    ψ = foldr(⊗, states)
    ψ = Vector{ComplexF64}(ψ)
    return ψ
end

@doc raw"""
    lift(U::AbstractMatrix{<:Number}, qubit_index::Int, n_qubits::Int; levels::Int=size(U, 1))

Lift an operator `U` acting on a single qubit to an operator acting on the entire system of `n_qubits`.
"""
function lift(
    U::AbstractMatrix{<:Number},
    qubit_index::Int,
    n_qubits::Int;
    levels::Int=size(U, 1)
)::Matrix{ComplexF64}
    Is = Matrix{Complex}[I(levels) for _ = 1:n_qubits]
    Is[qubit_index] = U
    return foldr(⊗, Is)
end

@doc raw"""
    lift(op::AbstractMatrix{<:Number}, i::Int, subsystem_levels::Vector{Int})

Lift an operator `op` acting on the i-th subsystem to an operator acting on the entire system with given subsystem levels.
"""
function lift(
    op::AbstractMatrix{<:Number},
    i::Int,
    subsystem_levels::Vector{Int}
)::Matrix{ComplexF64}
    @assert size(op, 1) == size(op, 2) == subsystem_levels[i] "Operator must be square and match dimension of subsystem i"

    Is = [collect(1.0 * typeof(op)(I, l, l)) for l ∈ subsystem_levels]
    Is[i] = op
    return kron(1.0, Is...)
end




"""
    quantum harmonic oscillator operators
"""

@doc raw"""
    annihilate(levels::Int)

Get the annihilation operator for a system with `levels` levels.
"""
function annihilate(levels::Int)::Matrix{ComplexF64}
    return diagm(1 => map(sqrt, 1:levels - 1))
end

@doc raw"""
    create(levels::Int)

Get the creation operator for a system with `levels` levels.
"""
function create(levels::Int)
    return collect(annihilate(levels)')
end

@doc raw"""
    number(levels::Int)

Get the number operator `n = a'a` for a system with `levels` levels.
"""
function number(levels::Int)
    return create(levels) * annihilate(levels)
end

@doc raw"""
    quad(levels::Int)

Get the operator `n(n - I)` for a system with `levels` levels.
"""
function quad(levels::Int)
    return number(levels) * (number(levels) - I(levels))
end

@doc raw"""
    cavity_state(level::Int, cavity_levels::Int)

Generate the state vector for a given `level` in a cavity system with `cavity_levels` levels.

# Arguments
- `level::Int`: The index of the desired level (must be between 0 and `cavity_levels` - 1).
- `cavity_levels::Int`: The total number of levels in the cavity system.

# Returns
- `Vector{ComplexF64}`: A state vector with `cavity_levels` elements where the element at `level + 1` is 1.0 and all other elements are 0.0.

# Throws
- `BoundsError`: If `level` is less than 0 or greater than or equal to `cavity_levels`.

# Examples
```julia
julia> cavity_state(2, 5)
5-element Vector{ComplexF64}:
 0.0 + 0.0im
 0.0 + 0.0im
 1.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
```
"""
function cavity_state(level::Int, cavity_levels::Int)
    if level < 0 || level >= cavity_levels
        throw(BoundsError("Invalid level index $level for a cavity with $cavity_levels levels."))
    end
    state = zeros(ComplexF64, cavity_levels)
    state[level + 1] = 1.
    return state
end


@doc raw"""
    multimode system utilities
"""

function multimode_state(ψ::String, transmon_levels::Int, cavity_levels::Int)
    @assert length(ψ) == 2

    @assert transmon_levels ∈ 2:4

    transmon_state = ψ[1]

    @assert transmon_state ∈ ['g', 'e']

    cavity_state = parse(Int, ψ[2])

    @assert cavity_state ∈ 0:cavity_levels - 2 "cavity state must be in [0, ..., cavity_levels - 2] (hightest cavity level is prohibited)"

    ψ_transmon = zeros(ComplexF64, transmon_levels)
    ψ_transmon[transmon_state == 'g' ? 1 : 2] = 1.0

    ψ_cavity = zeros(ComplexF64, cavity_levels)
    ψ_cavity[cavity_state + 1] = 1.0

    return ψ_transmon ⊗ ψ_cavity
end


"""
    isomporphism utilities
"""

@doc raw"""
    vec⁻¹(x::AbstractVector)

Convert a vector `x` into a square matrix. The length of `x` must be a perfect square.
"""
function vec⁻¹(x::AbstractVector)
    n = isqrt(length(x))
    return reshape(x, n, n)
end

@doc raw"""
    ket_to_iso(ψ)

Convert a ket vector `ψ` into a complex vector with real and imaginary parts.
"""
ket_to_iso(ψ) = [real(ψ); imag(ψ)]

@doc raw"""
    iso_to_ket(ψ̃)

Convert a complex vector `ψ̃` with real and imaginary parts into a ket vector.
"""
iso_to_ket(ψ̃) = ψ̃[1:div(length(ψ̃), 2)] + im * ψ̃[(div(length(ψ̃), 2) + 1):end]

@doc raw"""
    iso_vec_to_operator(Ũ⃗::AbstractVector{R}) where R <: Real

Convert a real vector `Ũ⃗` into a complex matrix representing an operator.
"""
function iso_vec_to_operator(Ũ⃗::AbstractVector{R}) where R <: Real
    Ũ⃗_dim = div(length(Ũ⃗), 2)
    N = Int(sqrt(Ũ⃗_dim))
    U = Matrix{Complex{R}}(undef, N, N)
    for i=0:N-1
        U[:, i+1] .=
            @view(Ũ⃗[i * 2N .+ (1:N)]) +
            one(R) * im * @view(Ũ⃗[i * 2N .+ (N+1:2N)])
    end
    return U
end

@doc raw"""
    iso_vec_to_iso_operator(Ũ⃗::AbstractVector{R}) where R <: Real

Convert a real vector `Ũ⃗` into a real matrix representing an isomorphism operator.
"""
function iso_vec_to_iso_operator(Ũ⃗::AbstractVector{R}) where R <: Real
    N = Int(sqrt(length(Ũ⃗) ÷ 2))
    Ũ = Matrix{R}(undef, 2N, 2N)
    U_real = Matrix{R}(undef, N, N)
    U_imag = Matrix{R}(undef, N, N)
    for i=0:N-1
        U_real[:, i+1] .= @view(Ũ⃗[i*2N .+ (1:N)])
        U_imag[:, i+1] .= @view(Ũ⃗[i*2N .+ (N+1:2N)])
    end
    Ũ[1:N, 1:N] .= U_real
    Ũ[1:N, (N + 1):end] .= -U_imag
    Ũ[(N + 1):end, 1:N] .= U_imag
    Ũ[(N + 1):end, (N + 1):end] .= U_real
    return Ũ
end

@doc raw"""
    operator_to_iso_vec(U::AbstractMatrix{<:Complex})

Convert a complex matrix `U` representing an operator into a real vector.
"""
function operator_to_iso_vec(U::AbstractMatrix{<:Complex})
    N = size(U,1)
    Ũ⃗ = Vector{Float64}(undef, N^2 * 2)
    for i=0:N-1
        Ũ⃗[i*2N .+ (1:N)] .= real(@view(U[:, i+1]))
        Ũ⃗[i*2N .+ (N+1:2N)] .= imag(@view(U[:, i+1]))
    end
    return Ũ⃗
end

@doc raw"""
    iso_operator_to_iso_vec(Ũ::AbstractMatrix{R}) where R <: Real

Convert a real matrix `Ũ` representing an isomorphism operator into a real vector.
"""
function iso_operator_to_iso_vec(Ũ::AbstractMatrix{R}) where R <: Real
    N = size(Ũ, 1) ÷ 2
    Ũ⃗ = Vector{R}(undef, N^2 * 2)
    for i=0:N-1
        Ũ⃗[i*2N .+ (1:2N)] .= @view Ũ[:, i+1]
    end
    return Ũ⃗
end


"""
    quantum metrics
"""

@doc raw"""
    fidelity(ψ, ψ_goal; subspace=1:length(ψ))

Calculate the fidelity between two quantum states `ψ` and `ψ_goal`.
"""
function fidelity(ψ, ψ_goal; subspace=1:length(ψ))
    ψ = ψ[subspace]
    ψ_goal = ψ_goal[subspace]
    return abs2(ψ_goal' * ψ)
end

@doc raw"""
    iso_fidelity(ψ̃, ψ̃_goal; kwargs...)

Calculate the fidelity between two quantum states in their isomorphic form `ψ̃` and `ψ̃_goal`.
"""
function iso_fidelity(ψ̃, ψ̃_goal; kwargs...)
    ψ = iso_to_ket(ψ̃)
    ψ_goal = iso_to_ket(ψ̃_goal)
    return fidelity(ψ, ψ_goal; kwargs...)
end

@doc raw"""
    unitary_fidelity(U::Matrix, U_goal::Matrix; subspace=nothing)

Calculate the fidelity between two unitary operators `U` and `U_goal`.
"""
function unitary_fidelity(
    U::Matrix,
    U_goal::Matrix;
    subspace=nothing
)
    if isnothing(subspace)
        N = size(U, 1)
        return 1 / N * abs(tr(U_goal'U))
    else
        U_goal = U_goal[subspace, subspace]
        U = U[subspace, subspace]
        N = length(subspace)
        return 1 / N * abs(tr(U_goal'U))
    end
end

@doc raw"""
    unitary_fidelity(Ũ⃗::AbstractVector{<:Real}, Ũ⃗_goal::AbstractVector{<:Real}; subspace=nothing)

Calculate the fidelity between two unitary operators in their isomorphic form `Ũ⃗` and `Ũ⃗_goal`.
"""
function unitary_fidelity(
    Ũ⃗::AbstractVector{<:Real},
    Ũ⃗_goal::AbstractVector{<:Real};
    subspace=nothing
)
    U = iso_vec_to_operator(Ũ⃗)
    U_goal = iso_vec_to_operator(Ũ⃗_goal)
    return unitary_fidelity(U, U_goal; subspace=subspace)
end

# TODO: add unitary squared fidelity

"""
    quantum measurement functions
"""

@doc raw"""
    population(ψ̃, i)

Calculate the population of the i-th level for a given state vector `ψ̃` in its isomorphic form.
"""
function population(ψ̃, i)
    @assert i ∈ 0:length(ψ̃) ÷ 2 - 1
    ψ = iso_to_ket(ψ̃)
    return abs2(ψ[i + 1])
end

@doc raw"""
    populations(ψ::AbstractVector{<:Complex})

Calculate the populations for each level of a given state vector `ψ`.
"""
function populations(ψ::AbstractVector{<:Complex})
    return abs2.(ψ)
end

@doc raw"""
    populations(ψ̃::AbstractVector{<:Real})

Calculate the populations for each level of a given state vector `ψ̃` in its isomorphic form.
"""
function populations(ψ̃::AbstractVector{<:Real})
    return populations(iso_to_ket(ψ̃))
end


@doc raw"""
    quantum_state(
        ket::String,
        levels::Vector{Int};
        level_dict=Dict(:g => 0, :e => 1, :f => 2, :h => 2),
        return_states=false
    )

Construct a quantum state from a string ket representation.

# Example

# TODO: add example
"""
function quantum_state(
    ket::String,
    levels::Vector{Int};
    level_dict=Dict(:g => 0, :e => 1, :f => 2, :h => 2),
    return_states=false
)
    kets = []

    for x ∈ split(ket, ['(', ')'])
        if x == ""
            continue
        elseif all(Symbol(xᵢ) ∈ keys(level_dict) for xᵢ ∈ x)
            append!(kets, x)
        elseif occursin("+", x)
            superposition = split(x, '+')
            @assert all(all(Symbol(xᵢ) ∈ keys(level_dict) for xᵢ ∈ x) for x ∈ superposition) "Invalid ket: $x"
            @assert length(superposition) == 2 "Only two states can be superposed for now"
            push!(kets, x)
        else
            error("Invalid ket: $x")
        end
    end

    states = []

    for (ψᵢ, l) ∈ zip(kets, levels)
        if ψᵢ isa AbstractString && occursin("+", ψᵢ)
            superposition = split(ψᵢ, '+')
            superposition_states = [level_dict[Symbol(x)] for x ∈ superposition]
            @assert all(state ≤ l - 1 for state ∈ superposition_states) "Level $ψᵢ is not allowed for $l levels"
            superposition_state = sum([
                cavity_state(state, l) for state ∈ superposition_states
            ])
            normalize!(superposition_state)
            push!(states, superposition_state)
        else
            state = level_dict[Symbol(ψᵢ)]
            @assert state ≤ l - 1 "Level $ψᵢ is not allowed for $l levels"
            push!(states, cavity_state(state, l))
        end
    end

    if return_states
        return states
    else
        return kron([1.0], states...)
    end
end











end
