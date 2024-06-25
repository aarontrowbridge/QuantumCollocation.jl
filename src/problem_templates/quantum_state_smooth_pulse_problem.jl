"""
    QuantumStateSmoothPulseProblem(
        system::AbstractQuantumSystem,
        ψ_init::Union{AbstractVector{<:Number}, Vector{<:AbstractVector{<:Number}}},
        ψ_goal::Union{AbstractVector{<:Number}, Vector{<:AbstractVector{<:Number}}},
        T::Int,
        Δt::Float64;
        kwargs...
    )

    QuantumStateSmoothPulseProblem(
        H_drift::AbstractMatrix{<:Number},
        H_drives::Vector{<:AbstractMatrix{<:Number}},
        args...;
        kwargs...
    )

Create a quantum control problem for smooth pulse optimization of a quantum state trajectory.

# Keyword Arguments

# TODO: clean up this whole constructor

"""
function QuantumStateSmoothPulseProblem end

function QuantumStateSmoothPulseProblem(
    system::AbstractQuantumSystem,
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
    leakage_indcies=nothing,
    integrator::Symbol=:pade,
    pade_order::Int=4,
    autodiff::Bool=pade_order != 4,
    rollout_integrator=exp,
    bound_state=integrator == :exponential,
    # TODO: control modulus norm, advanced feature, needs documentation
    control_norm_constraint=false,
    control_norm_constraint_components=nothing,
    control_norm_R=nothing,
    verbose=false,
    kwargs...
)
    @assert all(name ∈ L1_regularized_names for name in keys(L1_regularized_indices) if !isempty(L1_regularized_indices[name]))
    if !isnothing(a_guess)
        @assert size(a_guess) == (length(system.G_drives), T) "a_guess (size = $(size(a_guess))) must have size (length(system.G_drives), T)"
    end

    if ψ_init isa AbstractVector{<:Number} && ψ_goal isa AbstractVector{<:Number}
        ψ_inits = [ψ_init]
        ψ_goals = [ψ_goal]
    else
        @assert length(ψ_init) == length(ψ_goal)
        ψ_inits = ψ_init
        ψ_goals = ψ_goal
    end

    ψ_inits = Vector{ComplexF64}.(ψ_inits)
    ψ̃_inits = ket_to_iso.(ψ_inits)

    ψ_goals = Vector{ComplexF64}.(ψ_goals)
    ψ̃_goals = ket_to_iso.(ψ_goals)

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
                Symbol("ψ̃$i") => rollout(ψ̃_init, a_guess, Δt, system; integrator=rollout_integrator)
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

        if bound_state
            ψ̃_dim = size(ψ̃_inits[1], 1)
            ψ̃_bounds = NamedTuple([
                Symbol("ψ̃$i") => (-ones(ψ̃_dim), ones(ψ̃_dim))
                    for i = 1:length(ψ_inits)
            ])
        end

        if free_time

            control_components = (
                a = a,
                da = da,
                dda = dda,
                Δt = Δt,
            )

            components = merge(ψ̃s, control_components)

            controls = (:dda, :Δt)

            bounds = (
                a = a_bounds,
                dda = dda_bounds,
                Δt = (Δt_min, Δt_max),
            )

        else
            control_components = (
                a = a,
                da = da,
                dda = dda,
            )

            components = merge(ψ̃s, control_components)

            controls = (:dda,)

            bounds = (
                a = a_bounds,
                dda = dda_bounds,
            )

        end

        bounds = merge(bounds, ψ̃_bounds)

        traj = NamedTrajectory(
            components;
            controls=controls,
            timestep=free_time ? :Δt : Δt,
            bounds=bounds,
            initial=initial,
            final=final,
            goal=goal
        )
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
            J_L1, slack_con = L1Regularizer(
                name,
                traj;
                R_value=R_L1,
                indices=L1_regularized_indices[name]
            )
        else
            J_L1, slack_con = L1Regularizer(name, traj; R_value=R_L1)
        end
        J += J_L1
        push!(L1_slack_constraints, slack_con)
    end

    if !isnothing(leakage_indcies)
        for i = 1:length(ψ_inits)
            J_L1_ψ̃_i, slack_con_ψ̃_i = L1Regularizer(
                Symbol("ψ̃$i"),
                traj;
                R_value=R_L1,
                indices=leakage_indcies
            )
            J += J_L1_ψ̃_i
            push!(L1_slack_constraints, slack_con_ψ̃_i)
        end
    end

    append!(constraints, L1_slack_constraints)

    if integrator == :pade
        ψ̃_integrators = [
            QuantumStatePadeIntegrator(system, Symbol("ψ̃$i"), :a;
                order=pade_order,
                autodiff=autodiff,
            )
                for i = 1:length(ψ_inits)
        ]
    elseif integrator == :exponential
        ψ̃_integrators = [
            QuantumStateExponentialIntegrator(system, Symbol("ψ̃$i"), :a)
                for i = 1:length(ψ_inits)
        ]
    else
        error("integrator must be one of (:pade, :exponential)")
    end



    integrators = [
        ψ̃_integrators...,
        DerivativeIntegrator(:a, :da, traj),
        DerivativeIntegrator(:da, :dda, traj)
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
        kwargs...
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
