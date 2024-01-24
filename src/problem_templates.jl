module ProblemTemplates

export UnitarySmoothPulseProblem
export UnitaryMinimumTimeProblem
export UnitaryRobustnessProblem

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

@doc raw"""
    UnitarySmoothPulseProblem(H_drift, H_drives, U_goal, T, Δt; kwargs...)
    UnitarySmoothPulseProblem(system::QuantumSystem, U_goal, T, Δt; kwargs...)

Construct a `QuantumControlProblem` for a free-time unitary gate problem with smooth control pulses enforced by constraining the second derivative of the pulse trajectory, i.e.,

```math
\begin{aligned}
\underset{\vec{\tilde{U}}, a, \dot{a}, \ddot{a}, \Delta t}{\text{minimize}} & \quad
Q \cdot \ell\qty(\vec{\tilde{U}}_T, \vec{\tilde{U}}_{\text{goal}}) + \frac{1}{2} \sum_t \qty(R_a a_t^2 + R_{\dot{a}} \dot{a}_t^2 + R_{\ddot{a}} \ddot{a}_t^2) \\
\text{ subject to } & \quad \vb{P}^{(n)}\qty(\vec{\tilde{U}}_{t+1}, \vec{\tilde{U}}_t, a_t, \Delta t_t) = 0 \\
& a_{t+1} - a_t - \dot{a}_t \Delta t_t = 0 \\
& \quad \dot{a}_{t+1} - \dot{a}_t - \ddot{a}_t \Delta t_t = 0 \\
& \quad |a_t| \leq a_{\text{bound}} \\
& \quad |\ddot{a}_t| \leq \ddot{a}_{\text{bound}} \\
& \quad \Delta t_{\text{min}} \leq \Delta t_t \leq \Delta t_{\text{max}} \\
\end{aligned}
```

where, for $U \in SU(N)$,

```math
\ell\qty(\vec{\tilde{U}}_T, \vec{\tilde{U}}_{\text{goal}}) =
\abs{1 - \frac{1}{N} \abs{ \tr \qty(U_{\text{goal}}, U_T)} }
```

is the *infidelity* objective function, $Q$ is a weight, $R_a$, $R_{\dot{a}}$, and $R_{\ddot{a}}$ are weights on the regularization terms, and $\vb{P}^{(n)}$ is the $n$th-order Pade integrator.

# Arguments

- `H_drift::AbstractMatrix{<:Number}`: the drift hamiltonian
- `H_drives::Vector{<:AbstractMatrix{<:Number}}`: the control hamiltonians
or
- `system::QuantumSystem`: the system to be controlled
with
- `U_goal::AbstractMatrix{<:Number}`: the target unitary
- `T::Int`: the number of timesteps
- `Δt::Float64`: the (initial) time step size

# Keyword Arguments
- `free_time::Bool=true`: whether or not to allow the time steps to vary
- `init_trajectory::Union{NamedTrajectory, Nothing}=nothing`: an initial trajectory to use
- `a_bound::Float64=1.0`: the bound on the control pulse
- `a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives))`: the bounds on the control pulses, one for each drive
- `a_guess::Union{Matrix{Float64}, Nothing}=nothing`: an initial guess for the control pulses
- `dda_bound::Float64=1.0`: the bound on the control pulse derivative
- `dda_bounds::Vector{Float64}=fill(dda_bound, length(system.G_drives))`: the bounds on the control pulse derivatives, one for each drive
- `Δt_min::Float64=0.5 * Δt`: the minimum time step size
- `Δt_max::Float64=1.5 * Δt`: the maximum time step size
- `drive_derivative_σ::Float64=0.01`: the standard deviation of the initial guess for the control pulse derivatives
- `Q::Float64=100.0`: the weight on the infidelity objective
- `R=1e-2`: the weight on the regularization terms
- `R_a::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulses
- `R_da::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse derivatives
- `R_dda::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse second derivatives
- `leakage_suppression::Bool=false`: whether or not to suppress leakage to higher energy states
- `leakage_indices::Union{Nothing, Vector{Int}}=nothing`: the indices of $\vec{\tilde{U}}$ corresponding leakage operators that should be suppressed
- `system_levels::Union{Nothing, Vector{Int}}=nothing`: the number of levels in each subsystem
- `R_leakage=1e-1`: the weight on the leakage suppression term
- `max_iter::Int=1000`: the maximum number of iterations for the solver
- `linear_solver::String="mumps"`: the linear solver to use
- `ipopt_options::Options=Options()`: the options for the Ipopt solver
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: additional constraints to add to the problem
- `timesteps_all_equal::Bool=true`: whether or not to enforce that all time steps are equal
- `verbose::Bool=false`: whether or not to print constructor output
- `U_init::Union{AbstractMatrix{<:Number},Nothing}=nothing`: an initial guess for the unitary
- `integrator=Integrators.fourth_order_pade`: the integrator to use for the unitary
- `geodesic=true`: whether or not to use the geodesic as the initial guess for the unitary
- `pade_order=4`: the order of the Pade approximation to use for the unitary integrator
- `autodiff=pade_order != 4`: whether or not to use automatic differentiation for the unitary integrator
- `subspace=nothing`: the subspace to use for the unitary integrator
- `jacobian_structure=true`: whether or not to use the jacobian structure
- `hessian_approximation=false`: whether or not to use L-BFGS hessian approximation in Ipopt
- `blas_multithreading=true`: whether or not to use multithreading in BLAS
"""
function UnitarySmoothPulseProblem end

function UnitarySmoothPulseProblem(
    system::QuantumSystem,
    U_goal::AbstractMatrix{<:Number},
    T::Int,
    Δt::Float64;
    free_time=true,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives)),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    dda_bound::Float64=1.0,
    dda_bounds::Vector{Float64}=fill(dda_bound, length(system.G_drives)),
    Δt_min::Float64=0.5 * Δt,
    Δt_max::Float64=1.5 * Δt,
    drive_derivative_σ::Float64=0.01,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    leakage_suppression=false,
    leakage_indices=nothing,
    system_levels=nothing,
    R_leakage=1e-1,
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
    autodiff=pade_order != 4,
    subspace=nothing,
    jacobian_structure=true,
    hessian_approximation=false,
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

    if leakage_suppression
        if isnothing(leakage_indices)
            @assert !isnothing(system_levels) "if leakage_indices is not nothing, system_levels must be provided"
            leakage_indices = unitary_isomorphism_leakage_indices(system_levels)
        end
        J_leakage, slack_con = L1Regularizer(:Ũ⃗, traj; R_value=R_leakage, indices=leakage_indices)
        push!(constraints, slack_con)
        J += J_leakage
    end

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
    subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    hessian_approximation::Bool=true,
    D=1.0,
    verbose::Bool=false,
    ipopt_options::Options=Options(),
    kwargs...
)
    @assert unitary_symbol ∈ trajectory.names

    if hessian_approximation
        ipopt_options.hessian_approximation = "limited-memory"
    end

    objective += MinimumTimeObjective(trajectory; D=D)

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        unitary_symbol,
        final_fidelity,
        trajectory,
        subspace=subspace,
        hessian=!hessian_approximation
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
        hessian_approximation=hessian_approximation,
        kwargs...
    )
end

function UnitaryMinimumTimeProblem(
    prob::QuantumControlProblem;
    constraints::Union{Vector{<:AbstractConstraint}, Nothing}=nothing,
    objective::Union{Objective, Nothing}=nothing,
    build_trajectory_constraints::Bool=true,
    kwargs...
)
    params = deepcopy(prob.params)
    trajectory = copy(prob.trajectory)
    system = prob.system
    integrators = prob.integrators

    if isnothing(objective)
        objective = Objective(params[:objective_terms])
    end

    if isnothing(constraints)
        constraints = [
            params[:linear_constraints]...,
            NonlinearConstraint.(params[:nonlinear_constraints])...
        ]
        build_trajectory_constraints=false
    else
        build_trajectory_constraints=build_trajectory_constraints
    end

    return UnitaryMinimumTimeProblem(
        trajectory,
        system,
        objective,
        integrators,
        constraints;
        build_trajectory_constraints=build_trajectory_constraints,
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
    integrators = data["integrators"]
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

function UnitaryRobustnessProblem(
    Hₑ::AbstractMatrix{<:Number},
    trajectory::NamedTrajectory,
    system::QuantumSystem,
    objective::Objective,
    integrators::Vector{<:AbstractIntegrator},
    constraints::Vector{<:AbstractConstraint};
    unitary_symbol::Symbol=:Ũ⃗,
    final_fidelity::Float64=unitary_fidelity(trajectory[end][unitary_symbol], trajectory.goal[unitary_symbol]),
    subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    hessian_approximation::Bool=true,
    verbose::Bool=false,
    ipopt_options::Options=Options(),
    kwargs...
)
    @assert unitary_symbol ∈ trajectory.names

    if hessian_approximation
        ipopt_options.hessian_approximation = "limited-memory"
    end

    objective += InfidelityRobustnessObjective(
        Hₑ,
        trajectory,
        eval_hessian=!hessian_approximation,
        subspace=subspace
    )

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        unitary_symbol,
        final_fidelity,
        trajectory;
        subspace=subspace,
        hessian=!hessian_approximation
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
        hessian_approximation=hessian_approximation,
        kwargs...
    )
end

function UnitaryRobustnessProblem(
    Hₑ::AbstractMatrix{<:Number},
    prob::QuantumControlProblem;
    constraints::Union{Vector{<:AbstractConstraint}, Nothing}=nothing,
    objective::Union{Objective, Nothing}=nothing,
    build_trajectory_constraints::Bool=true,
    kwargs...
)
    params = deepcopy(prob.params)
    trajectory = copy(prob.trajectory)
    system = prob.system
    integrators = prob.integrators

    if isnothing(objective)
        objective = Objective(params[:objective_terms])
    end

    if isnothing(constraints)
        constraints = [
            params[:linear_constraints]...,
            NonlinearConstraint.(params[:nonlinear_constraints])...
        ]
        build_trajectory_constraints=false
    else
        build_trajectory_constraints=build_trajectory_constraints
    end

    return UnitaryRobustnessProblem(
        Hₑ,
        trajectory,
        system,
        objective,
        integrators,
        constraints;
        build_trajectory_constraints=build_trajectory_constraints,
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
