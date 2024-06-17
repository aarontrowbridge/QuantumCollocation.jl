module QuantumSystemUtils

export operator_algebra
export is_reachable

using ..EmbeddedOperators
using ..QuantumSystems

using LinearAlgebra
using SparseArrays


commutator(A::AbstractMatrix, B::AbstractMatrix) = A * B - B * A

# TODO: Any difference between LinearAlgebra::ishermitian?
is_hermitian(H::AbstractMatrix; atol=eps(Float32)) =
    all(isapprox.(H - H', 0.0, atol=atol))

function is_linearly_dependent(basis::Vector{<:AbstractMatrix}; kwargs...)
    return is_linearly_dependent(stack(vec.(basis)); kwargs...)
end

function is_linearly_dependent(M::AbstractMatrix; eps=eps(Float32), verbose=true)
    if size(M, 2) > size(M, 1)
        if verbose
            println("Linearly dependent because columns > rows, $(size(M, 2)) > $(size(M, 1)).")
        end
        return true
    end
    # QR decomposition has a zero R on diagonal if linearly dependent
    val = minimum(abs.(diag(qr(M).R)))
    return isapprox(val, 0.0, atol=eps)
end

function linearly_independent_indices(
    basis::Vector{<:AbstractMatrix};
    order=1:length(basis),
    kwargs...
)
    @assert issetequal(order, 1:length(basis)) "Order must enumerate entire basis."
    bᵢ = Int[]
    for i ∈ order
        if !is_linearly_dependent([basis[bᵢ]..., basis[i]]; kwargs...)
            push!(bᵢ, i)
        end
    end
    return bᵢ
end

function linearly_independent_subset(basis::Vector{<:AbstractMatrix}; kwargs...)
    bᵢ = linearly_independent_indices(basis; kwargs...)
    return deepcopy(basis[bᵢ])
end

function linearly_independent_subset!(basis::Vector{<:AbstractMatrix}; kwargs...)
    bᵢ = linearly_independent_indices(basis; kwargs...)
    deleteat!(basis, setdiff(1:length(basis), bᵢ))
    return nothing
end

traceless(M::AbstractMatrix) = M - tr(M) * I / size(M, 1)

"""
operator_algebra(generators; return_layers=false, normalize=false, verbose=false, remove_trace=true)

    Compute the Lie algebra basis for the given generators.
    If return_layers is true, the Lie tree layers are also returned.

    Can normalize the basis and enforce traceless generators.

    # Arguments
    - `generators::Vector{<:AbstractMatrix}`: generators of the Lie algebra
    - `return_layers::Bool=false`: return the Lie tree layers
    - `normalize::Bool=false`: normalize the basis
    - `verbose::Bool=false`: print debug information
    - `remove_trace::Bool=true`: remove trace from generators
"""
function operator_algebra(
    generators::Vector{<:AbstractMatrix{T}}; 
    return_layers=false, 
    normalize=false,
    verbose=false,
    remove_trace=true
) where T<:Number
    # Initialize basis (traceless, normalized)
    basis = normalize ? [g / norm(g) for g ∈ generators] : deepcopy(generators)
    if remove_trace
        @. basis = traceless(basis)
    end

    # Initialize layers
    current_layer = deepcopy(basis)
    if return_layers
        all_layers = Vector{Matrix{T}}[deepcopy(basis)]
    end

    ℓ = 1
    if verbose
        println("ℓ = $ℓ")
    end
    if is_linearly_dependent(basis)
        println("Linearly dependent generators.")
    else
        # Note: Use left normalized commutators
        # Note: Jacobi identity is not checked
        need_basis = true
        algebra_dim = size(first(generators), 1)^2 - 1
        while need_basis
            layer = Matrix{T}[]
            # Repeat commutators until no new operators are found
            for op ∈ current_layer
                for gen ∈ generators
                    if !need_basis
                        continue
                    end

                    test = commutator(gen, op)
                    if all(test .≈ 0)
                        continue
                    # Current basis is assumed to be linearly independent
                    elseif is_linearly_dependent([basis..., test], verbose=verbose)
                        continue
                    else
                        test .= is_hermitian(test) ? test : im * test
                        test .= normalize ? test / norm(test) : test
                        push!(basis, test)
                        push!(layer, test)
                        need_basis = length(basis) < algebra_dim ? true : false
                    end
                end
            end

            if isempty(layer)
                println("Subspace termination.")
                break
            else
                current_layer = layer
                ℓ += 1
                if verbose
                    println("ℓ = $ℓ")
                end
            end

            if return_layers
                append!(all_layers, [current_layer])
            end
        end
    end

    if return_layers
        return basis, all_layers
    else
        return basis
    end
end

function fit_gen_to_basis(
    gen::AbstractMatrix{<:T},
    basis::AbstractVector{<:AbstractMatrix{<:T}}
) where T<:Number
    A = stack(vec.(basis))
    b = vec(gen)
    return A \ b
end

function is_in_span(
    gen::AbstractMatrix,
    basis::AbstractVector{<:AbstractMatrix};
    subspace_indices::AbstractVector{<:Int}=1:size(gen, 1),
    atol=eps(Float32),
    return_effective_gen=false,
)
    g_basis = [deepcopy(b[subspace_indices, subspace_indices]) for b ∈ basis]
    linearly_independent_subset!(g_basis)
    # Traceless basis needs traceless fit
    x = fit_gen_to_basis(gen, g_basis)
    g_eff = sum(x .* g_basis)
    ε = norm(g_eff - gen, 2)
    if return_effective_gen
        return ε < atol, g_eff
    else
        return ε < atol
    end
end

"""
is_reachable(hamiltonians, gate, subspace_levels, levels; kwargs...)

    Check if the gate is reachable from the given generators.

    # Arguments
    - `hamiltonians::AbstractVector{<:AbstractMatrix}`: generators of the Lie algebra
    - `gate::AbstractMatrix`: target gate
    - `subspace_levels::Vector{<:Int}`: levels of the subspace
    - `levels::Vector{<:Int}`: levels of the system
"""
function is_reachable(
    gate::AbstractMatrix,
    hamiltonians::AbstractVector{<:AbstractMatrix};
    subspace_indices::AbstractVector{<:Int}=1:size(gate, 1),
    compute_basis=true,
    remove_trace=true,
    verbose=false,
    atol=eps(Float32)
)
    generator = im * log(gate)

    if remove_trace
        generator = traceless(generator)
    end

    if compute_basis
        basis = operator_algebra(hamiltonians, remove_trace=remove_trace, verbose=verbose)
    else
        basis = hamiltonians
    end

    return is_in_span(
        generator,
        basis,
        subspace_indices=subspace_indices,
        atol=atol
    )
end

function is_reachable(
    gate::AbstractMatrix,
    system::QuantumSystem;
    use_drift::Bool=true,
    kwargs...
)
    if !iszero(system.H_drift) && use_drift
        hamiltonians = [system.H_drift, system.H_drives...]
    else
        hamiltonians = system.H_drives
    end
    return is_reachable(gate, hamiltonians; kwargs...)
end


end