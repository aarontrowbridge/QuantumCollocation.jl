module ProblemTemplates

export UnitarySmoothPulseProblem
export UnitaryMinimumTimeProblem

export QuantumStateSmoothPulseProblem
export QuantumStateMinimumTimeProblem

using ..QuantumSystems
using ..QuantumUtils
using ..Rollouts
using ..Objectives
using ..Constraints
using ..Integrators
using ..Problems
using ..IpoptOptions

using NamedTrajectories
using LinearAlgebra
using Distributions
using JLD2


# -------------------------------------------
# Unitary Problem Templates
# -------------------------------------------

function UnitarySmoothPulseProblem(
    system::QuantumSystem,
    U_goal::AbstractMatrix{<:Number},
    T::Int,
    Δt::Float64;
    free_time=true,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=Inf,
    a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    dda_bound::Float64=Inf,
    dda_bounds::Vector{Float64}=fill(dda_bound, length(system.G_drives)),
    Δt_min::Float64=0.5 * Δt,
    Δt_max::Float64=1.5 * Δt,
    drive_derivative_σ::Float64=0.01,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    max_iter::Int=1000,
    linear_solver::String="mumps",
    ipopt_options::Options=Options(),
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    timesteps_all_equal::Bool=true,
    verbose::Bool=false,
    U_init::Union{AbstractMatrix{<:Number},Nothing}=nothing,
    integrator=Integrators.fourth_order_pade,
    geodesic=true,
    pade_order=4,
    autodiff=false,
    subspace=nothing,
    jacobian_structure=true,
    hessian_approximation=false,
    jacobian_chunk_size=10,
    blas_multithreading=true,
)
    U_goal = Matrix{ComplexF64}(U_goal)

    if !blas_multithreading
        BLAS.set_num_threads(1)
    end

    if hessian_approximation
        ipopt_options.hessian_approximation = "limited-memory"
    end

    if isnothing(U_init)
        Ũ⃗_init = operator_to_iso_vec(1.0I(size(U_goal, 1)))
    else
        Ũ⃗_init = operator_to_iso_vec(U_init)
    end

    n_drives = length(system.G_drives)

    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        if free_time
            Δt = fill(Δt, 1, T)
        end

        if isnothing(a_guess)
            geodesic_success = true
            if geodesic
                try
                    Ũ⃗ = unitary_geodesic(U_goal, T)
                catch e
                    @warn "Could not find geodesic. Using random initial guess."
                    geodesic_success = false
                end
            end
            if !geodesic || !geodesic_success
                Ũ⃗ = 2 * rand(length(Ũ⃗_init), T) .- 1
            end
            a_dists =  [Uniform(-a_bounds[i], a_bounds[i]) for i = 1:n_drives]
            a = hcat([
                zeros(n_drives),
                vcat([rand(a_dists[i], 1, T - 2) for i = 1:n_drives]...),
                zeros(n_drives)
            ]...)

            da = randn(n_drives, T) * drive_derivative_σ
            dda = randn(n_drives, T) * drive_derivative_σ
        else
            Ũ⃗ = unitary_rollout(Ũ⃗_init, a_guess, Δt, system; integrator=integrator)
            a = a_guess
            da = derivative(a, Δt)
            dda = derivative(da, Δt)
        end

        if isnothing(U_init)
            Ũ⃗_init = operator_to_iso_vec(1.0I(size(U_goal, 1)))
        else
            Ũ⃗_init = operator_to_iso_vec(U_init)
        end

        initial = (
            Ũ⃗ = Ũ⃗_init,
            a = zeros(n_drives),
        )

        final = (
            a = zeros(n_drives),
        )

        goal = (
            Ũ⃗ = operator_to_iso_vec(U_goal),
        )

        if free_time
            components = (
                Ũ⃗ = Ũ⃗,
                a = a,
                da = da,
                dda = dda,
                Δt = Δt,
            )

            bounds = (
                a = a_bounds,
                dda = dda_bounds,
                Δt = (Δt_min, Δt_max),
            )

            traj = NamedTrajectory(
                components;
                controls=(:dda, :Δt),
                timestep=:Δt,
                bounds=bounds,
                initial=initial,
                final=final,
                goal=goal
            )
        else
            components = (
                Ũ⃗ = Ũ⃗,
                a = a,
                da = da,
                dda = dda,
            )

            bounds = (
                a = a_bounds,
                dda = dda_bounds,
            )

            traj = NamedTrajectory(
                components;
                controls=(:dda,),
                timestep=Δt,
                bounds=bounds,
                initial=initial,
                final=final,
                goal=goal
            )
        end
    end

    J = UnitaryInfidelityObjective(:Ũ⃗, traj, Q; subspace=subspace)
    J += QuadraticRegularizer(:a, traj, R_a)
    J += QuadraticRegularizer(:da, traj, R_da)
    J += QuadraticRegularizer(:dda, traj, R_dda)

    integrators = [
        UnitaryPadeIntegrator(system, :Ũ⃗, :a; order=pade_order, autodiff=autodiff),
        DerivativeIntegrator(:a, :da, traj),
        DerivativeIntegrator(:da, :dda, traj),
    ]

    if free_time
        if timesteps_all_equal
            push!(constraints, TimeStepsAllEqualConstraint(:Δt, traj))
        end
    end

    return QuantumControlProblem(
        system,
        traj,
        J,
        integrators;
        constraints=constraints,
        max_iter=max_iter,
        linear_solver=linear_solver,
        verbose=verbose,
        ipopt_options=ipopt_options,
        jacobian_structure=jacobian_structure,
        hessian_approximation=hessian_approximation,
        jacobian_chunk_size=jacobian_chunk_size,
        eval_hessian=!hessian_approximation
    )
end

function UnitarySmoothPulseProblem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    args...;
    kwargs...
)
    system = QuantumSystem(H_drift, H_drives)
    return UnitarySmoothPulseProblem(system, args...; kwargs...)
end

function UnitaryMinimumTimeProblem(
    trajectory::NamedTrajectory,
    system::QuantumSystem,
    objective::Objective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint};
    unitary_symbol::Symbol=:Ũ⃗,
    final_fidelity::Float64=unitary_fidelity(trajectory[end][unitary_symbol], trajectory.goal[unitary_symbol]),
    D=1.0,
    verbose::Bool=false,
    ipopt_options::Options=Options(),
    kwargs...
)
    @assert unitary_symbol ∈ trajectory.names

    objective += MinimumTimeObjective(trajectory; D=D)

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        unitary_symbol,
        final_fidelity,
        trajectory
    )

    constraints = AbstractConstraint[constraints..., fidelity_constraint]

    return QuantumControlProblem(
        system,
        trajectory,
        objective,
        integrators;
        constraints=constraints,
        verbose=verbose,
        ipopt_options=ipopt_options,
        kwargs...
    )
end

function UnitaryMinimumTimeProblem(
    prob::QuantumControlProblem;
    kwargs...
)
    params = deepcopy(prob.params)
    traj = copy(prob.trajectory)
    system = prob.system
    objective = Objective(params[:objective_terms])
    integrators = prob.integrators
    constraints = [
        params[:linear_constraints]...,
        NonlinearConstraint.(params[:nonlinear_constraints])...
    ]
    return UnitaryMinimumTimeProblem(
        traj,
        system,
        objective,
        integrators,
        constraints;
        build_trajectory_constraints=false,
        kwargs...
    )
end


function UnitaryMinimumTimeProblem(
    data_path::String;
    kwargs...
)
    data = load(data_path)
    system = data["system"]
    trajectory = data["trajectory"]
    objective = Objective(data["params"][:objective_terms])
    integrators = data["params"][:dynamics]
    constraints = AbstractConstraint[
        data["params"][:linear_constraints]...,
        NonlinearConstraint.(data["params"][:nonlinear_constraints])...
    ]
    return UnitaryMinimumTimeProblem(
        trajectory,
        system,
        objective,
        integrators,
        constraints;
        build_trajectory_constraints=false,
        kwargs...
    )
end

# ------------------------------------------
# Quantum State Problem Templates
# ------------------------------------------

function QuantumStateSmoothPulseProblem(
    system::QuantumSystem,
    ψ_init::Union{AbstractVector{<:Number}, Vector{<:AbstractVector{<:Number}}},
    ψ_goal::Union{AbstractVector{<:Number}, Vector{<:AbstractVector{<:Number}}},
    T::Int,
    Δt::Float64;
    free_time=true,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=Inf,
    a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    dda_bound::Float64=Inf,
    dda_bounds::Vector{Float64}=fill(dda_bound, length(system.G_drives)),
    Δt_min::Float64=0.5 * Δt,
    Δt_max::Float64=1.5 * Δt,
    drive_derivative_σ::Float64=0.01,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    R_L1::Float64=20.0,
    max_iter::Int=1000,
    linear_solver::String="mumps",
    ipopt_options::Options=Options(),
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    timesteps_all_equal::Bool=true,
    L1_regularized_names=Symbol[],
    L1_regularized_indices::NamedTuple=NamedTuple(),
    verbose=false,
)
    @assert all(name ∈ L1_regularized_names for name in keys(L1_regularized_indices) if !isempty(L1_regularized_indices[name]))

    if ψ_init isa AbstractVector{<:Number} && ψ_goal isa AbstractVector{<:Number}
        ψ_inits = [ψ_init]
        ψ_goals = [ψ_goal]
    else
        @assert length(ψ_init) == length(ψ_goal)
        ψ_inits = ψ_init
        ψ_goals = ψ_goal
    end

    ψ_inits = Vector{ComplexF64}.(ψ_init)
    ψ̃_inits = ket_to_iso.(ψ_init)

    ψ_goals = Vector{ComplexF64}.(ψ_goal)
    ψ̃_goals = ket_to_iso.(ψ_goal)

    n_drives = length(system.G_drives)

    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        if free_time
            Δt = fill(Δt, T)
        end

        if isnothing(a_guess)
            ψ̃s = NamedTuple([
                Symbol("ψ̃$i") => linear_interpolation(ψ̃_init, ψ̃_goal, T)
                    for (i, (ψ̃_init, ψ̃_goal)) in enumerate(zip(ψ̃_inits, ψ̃_goals))
            ])
            a_dists =  [Uniform(-a_bounds[i], a_bounds[i]) for i = 1:n_drives]
            a = hcat([
                zeros(n_drives),
                vcat([rand(a_dists[i], 1, T - 2) for i = 1:n_drives]...),
                zeros(n_drives)
            ]...)
            da = randn(n_drives, T) * drive_derivative_σ
            dda = randn(n_drives, T) * drive_derivative_σ
        else
            ψ̃s = NamedTuple([
                Symbol("ψ̃$i") => rollout(ψ̃_init, a_guess, Δt, system)
                    for (i, ψ̃_init) in enumerate(ψ̃_inits)
            ])
            a = a_guess
            da = derivative(a, Δt)
            dda = derivative(da, Δt)
        end

        ψ̃_initial = NamedTuple([
            Symbol("ψ̃$i") => ψ̃_init
                for (i, ψ̃_init) in enumerate(ψ̃_inits)
        ])

        control_initial = (
            a = zeros(n_drives),
        )

        initial = merge(ψ̃_initial, control_initial)

        final = (
            a = zeros(n_drives),
        )

        goal = NamedTuple([
            Symbol("ψ̃$i") => ψ̃_goal
                for (i, ψ̃_goal) in enumerate(ψ̃_goals)
        ])

        if free_time

            control_components = (
                a = a,
                da = da,
                dda = dda,
                Δt = Δt,
            )

            components = merge(ψ̃s, control_components)

            bounds = (
                a = a_bounds,
                dda = dda_bounds,
                Δt = (Δt_min, Δt_max),
            )

            traj = NamedTrajectory(
                components;
                controls=(:dda, :Δt),
                timestep=:Δt,
                bounds=bounds,
                initial=initial,
                final=final,
                goal=goal
            )
        else
            control_components = (
                a = a,
                da = da,
                dda = dda,
            )

            components = merge(ψ̃s, control_components)

            bounds = (
                a = a_bounds,
                dda = dda_bounds,
            )

            traj = NamedTrajectory(
                components;
                controls=(:dda,),
                timestep=Δt,
                bounds=bounds,
                initial=initial,
                final=final,
                goal=goal
            )
        end
    end

    J = QuadraticRegularizer(:a, traj, R_a)
    J += QuadraticRegularizer(:da, traj, R_da)
    J += QuadraticRegularizer(:dda, traj, R_dda)

    for i = 1:length(ψ_inits)
        J += QuantumStateObjective(Symbol("ψ̃$i"), traj, Q)
    end

    L1_slack_constraints = []

    for name in L1_regularized_names
        if name in keys(L1_regularized_indices)
            J_L1, slack_con = L1Regularizer(name, traj; R_value=R_L1, indices=L1_regularized_indices[name])
        else
            J_L1, slack_con = L1Regularizer(name, traj; R_value=R_L1)
        end
        J += J_L1
        push!(L1_slack_constraints, slack_con)
    end

    append!(constraints, L1_slack_constraints)

    if free_time

        ψ̃_integrators = [
            QuantumStatePadeIntegrator(system, Symbol("ψ̃$i"), :a)
                for i = 1:length(ψ_inits)
        ]

        integrators = [
            ψ̃_integrators...,
            DerivativeIntegrator(:a, :da, traj),
            DerivativeIntegrator(:da, :dda, traj)
        ]
    else
        ψ̃_integrators = [
            QuantumStatePadeIntegrator(system, Symbol("ψ̃$i"), :a)
                for i = 1:length(ψ_inits)
        ]

        integrators = [
            ψ̃_integrators...,
            DerivativeIntegrator(:a, :da, traj),
            DerivativeIntegrator(:da, :dda, traj)
        ]
    end

    if free_time
        if timesteps_all_equal
            push!(constraints, TimeStepsAllEqualConstraint(:Δt, traj))
        end
    end

    return QuantumControlProblem(
        system,
        traj,
        J,
        integrators;
        constraints=constraints,
        max_iter=max_iter,
        linear_solver=linear_solver,
        verbose=verbose,
        ipopt_options=ipopt_options,
    )
end

function QuantumStateSmoothPulseProblem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    args...;
    kwargs...
)
    system = QuantumSystem(H_drift, H_drives)
    return QuantumStateSmoothPulseProblem(system, args...; kwargs...)
end


function QuantumStateMinimumTimeProblem(
    trajectory::NamedTrajectory,
    system::QuantumSystem,
    objective::Objective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint};
    state_symbol::Symbol=:ψ̃,
    D=1.0,
    verbose::Bool=false,
    ipopt_options::Options=Options(),
    kwargs...
)
    @assert state_symbol ∈ trajectory.names

    objective += MinimumTimeObjective(trajectory; D=D)

    final_fidelity = fidelity(trajectory[end][state_symbol], trajectory.goal[state_symbol])

    fidelity_constraint = FinalQuantumStateFidelityConstraint(
        state_symbol,
        final_fidelity,
        trajectory
    )

    push!(constraints, fidelity_constraint)

    return QuantumControlProblem(
        system,
        trajectory,
        objective,
        integrators;
        constraints=constraints,
        verbose=verbose,
        ipopt_options=ipopt_options,
        kwargs...
    )
end

function QuantumStateMinimumTimeProblem(
    data_path::String;
    kwargs...
)
    data = load(data_path)
    system = data["system"]
    trajectory = data["trajectory"]
    objective = Objective(data["params"][:objective_terms])
    integrators = data["params"][:dynamics]
    constraints = AbstractConstraint[
        data["params"][:linear_constraints]...,
        NonlinearConstraint.(data["params"][:nonlinear_constraints])...
    ]
    return QuantumStateMinimumTimeProblem(
        trajectory,
        system,
        objective,
        integrators,
        constraints;
        build_trajectory_constraints=false,
        kwargs...
    )
end



end
