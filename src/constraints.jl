module Constraints

export constrain!
export trajectory_constraints

export AbstractConstraint

export NonlinearConstraint

export NonlinearEqualityConstraint

export NonlinearInequalityConstraint

export FinalFidelityConstraint
export FinalUnitaryFidelityConstraint
export FinalQuantumStateFidelityConstraint

export ComplexModulusContraint

export LinearConstraint

export EqualityConstraint
export BoundsConstraint
export TimeStepBoundsConstraint
export TimeStepEqualityConstraint
export TimeStepsAllEqualConstraint
export L1SlackConstraint

using ..StructureUtils
using ..QuantumUtils

using TrajectoryIndexingUtils
using NamedTrajectories
using ForwardDiff
using SparseArrays
using Ipopt
using MathOptInterface
const MOI = MathOptInterface


abstract type AbstractConstraint end

abstract type NonlinearConstraint <: AbstractConstraint end


function NonlinearConstraint(params::Dict)
    return eval(params[:type])(; delete!(params, :type)...)
end

"""
    struct NonlinearEqualityConstraint

Represents a nonlinear equality constraint.

# Fields
- `g::Function`: the constraint function
- `∂g::Function`: the Jacobian of the constraint function
- `∂g_structure::Vector{Tuple{Int, Int}}`: the structure of the Jacobian
   i.e. all non-zero entries
- `μ∂²g::Function`: the Hessian of the constraint function
- `μ∂²g_structure::Vector{Tuple{Int, Int}}`: the structure of the Hessian
- `dim::Int`: the dimension of the constraint function
- `params::Dict{Symbol, Any}`: a dictionary of parameters

"""
struct NonlinearEqualityConstraint <: NonlinearConstraint
    g::Function
    ∂g::Function
    ∂g_structure::Vector{Tuple{Int, Int}}
    μ∂²g::Function
    μ∂²g_structure::Vector{Tuple{Int, Int}}
    dim::Int
    params::Dict{Symbol, Any}
end

"""
    struct NonlinearInequalityConstraint

Represents a nonlinear inequality constraint.

# Fields
- `g::Function`: the constraint function
- `∂g::Function`: the Jacobian of the constraint function
- `∂g_structure::Vector{Tuple{Int, Int}}`: the structure of the Jacobian
   i.e. all non-zero entries
- `μ∂²g::Function`: the Hessian of the constraint function
- `μ∂²g_structure::Vector{Tuple{Int, Int}}`: the structure of the Hessian
- `dim::Int`: the dimension of the constraint function
- `params::Dict{Symbol, Any}`: a dictionary of parameters containing additional
   information about the constraint

"""
struct NonlinearInequalityConstraint <: NonlinearConstraint
    g::Function
    ∂g::Function
    ∂g_structure::Vector{Tuple{Int, Int}}
    μ∂²g::Function
    μ∂²g_structure::Vector{Tuple{Int, Int}}
    dim::Int
    params::Dict{Symbol, Any}
end

"""
    FinalFidelityConstraint(<keyword arguments>)


Returns a NonlinearInequalityConstraint representing a constraint on the
minimum allowed fidelity.

# Arguments
- `fidelity_function::Union{Function,Nothing}=nothing`: the fidelity function
- `value::Union{Float64,Nothing}=nothing`: the minimum fidelity value allowed
   by the constraint
- `comps::Union{AbstractVector{Int},Nothing}=nothing`: the components of the
   state to which the fidelity function is applied
- `goal::Union{AbstractVector{Float64},Nothing}=nothing`: the goal state
- `statedim::Union{Int,Nothing}=nothing`: the dimension of the state
- `zdim::Union{Int,Nothing}=nothing`: the dimension of a single time step of the trajectory
- `T::Union{Int,Nothing}=nothing`: the number of time steps
- `subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing`: the subspace indices of the fidelity
- `hessian::Bool=true`: whether or not to include the Hessian of the constraint
"""
function FinalFidelityConstraint(;
    fidelity_function::Union{Function,Nothing}=nothing,
    value::Union{Float64,Nothing}=nothing,
    comps::Union{AbstractVector{Int},Nothing}=nothing,
    goal::Union{AbstractVector{Float64},Nothing}=nothing,
    statedim::Union{Int,Nothing}=nothing,
    zdim::Union{Int,Nothing}=nothing,
    T::Union{Int,Nothing}=nothing,
    subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    hessian::Bool=true
)
    @assert !isnothing(fidelity_function) "must provide a fidelity function"
    @assert !isnothing(value) "must provide a fidelity value"
    @assert !isnothing(comps) "must provide a list of components"
    @assert !isnothing(goal) "must provide a goal state"
    @assert !isnothing(statedim) "must provide a state dimension"
    @assert !isnothing(zdim) "must provide a z dimension"
    @assert !isnothing(T) "must provide a T"

    fidelity_function_symbol = Symbol(fidelity_function)

    if isnothing(subspace)
        fid = x -> fidelity_function(x, goal)
    else
        fid = x -> fidelity_function(x, goal; subspace=subspace)
    end

    @assert fid(randn(statedim)) isa Float64 "fidelity function must return a scalar"

    params = Dict{Symbol, Any}()

    if fidelity_function_symbol ∉ names(QuantumUtils)
        @warn "fidelity function is not exported by QuantumUtils: will not be able to save this constraint"
        params[:type] = :FinalFidelityConstraint
        params[:fidelity_function] = :not_saveable
    else
        params[:type] = :FinalFidelityConstraint
        params[:fidelity_function] = fidelity_function_symbol
        params[:value] = value
        params[:comps] = comps
        params[:statedim] = statedim
        params[:zdim] = zdim
        params[:T] = T
        params[:subspace] = subspace
    end

    state_slice = slice(T, comps, zdim)

    ℱ(x) = [fid(x)]

    g(Z⃗) = ℱ(Z⃗[state_slice]) .- value

    ∂ℱ(x) = ForwardDiff.jacobian(ℱ, x)

    ∂ℱ_structure = jacobian_structure(∂ℱ, statedim)

    col_offset = index(T, comps[1] - 1, zdim)

    ∂g_structure = [(i, j + col_offset) for (i, j) in ∂ℱ_structure]

    @views function ∂g(Z⃗; ipopt=true)
        ∂ = spzeros(1, T * zdim)
        ∂ℱ_x = ∂ℱ(Z⃗[state_slice])
        for (i, j) ∈ ∂ℱ_structure
            ∂[i, j + col_offset] = ∂ℱ_x[i, j]
        end
        if ipopt
            return [∂[i, j] for (i, j) in ∂g_structure]
        else
            return ∂
        end
    end

    if hessian
        ∂²ℱ(x) = ForwardDiff.hessian(fid, x)

        ∂²ℱ_structure = hessian_of_lagrangian_structure(∂²ℱ, statedim, 1)

        μ∂²g_structure = [ij .+ col_offset for ij in ∂²ℱ_structure]

        @views function μ∂²g(Z⃗, μ; ipopt=true)
            HoL = spzeros(T * zdim, T * zdim)
            μ∂²ℱ = μ[1] * ∂²ℱ(Z⃗[state_slice])
            for (i, j) ∈ ∂²ℱ_structure
                HoL[i + col_offset, j + col_offset] = μ∂²ℱ[i, j]
            end
            if ipopt
                return [HoL[i, j] for (i, j) in μ∂²g_structure]
            else
                return HoL
            end
        end
    else
        μ∂²g_structure = nothing
        μ∂²g = nothing
    end

    return NonlinearInequalityConstraint(
        g,
        ∂g,
        ∂g_structure,
        μ∂²g,
        μ∂²g_structure,
        1,
        params
    )
end

"""
    FinalUnitaryFidelityConstraint(statesymb::Symbol, val::Float64, traj::NamedTrajectory)

Returns a FinalFidelityConstraint for the unitary fidelity function where statesymb
is the NamedTrajectory symbol representing the unitary.

"""
function FinalUnitaryFidelityConstraint(
    statesymb::Symbol,
    val::Float64,
    traj::NamedTrajectory;
    subspace::Union{AbstractVector{<:Integer}, Nothing}=nothing,
    hessian=true
)
    @assert statesymb ∈ traj.names
    return FinalFidelityConstraint(;
        fidelity_function=unitary_fidelity,
        value=val,
        comps=traj.components[statesymb],
        goal=traj.goal[statesymb],
        statedim=traj.dims[statesymb],
        zdim=traj.dim,
        T=traj.T,
        subspace=subspace,
        hessian=hessian
    )
end

"""
    FinalQuantumStateFidelityConstraint(statesymb::Symbol, val::Float64, traj::NamedTrajectory)

Returns a FinalFidelityConstraint for the unitary fidelity function where statesymb
is the NamedTrajectory symbol representing the unitary.

"""
function FinalQuantumStateFidelityConstraint(
    statesymb::Symbol,
    val::Float64,
    traj::NamedTrajectory
)
    @assert statesymb ∈ traj.names
    return FinalFidelityConstraint(;
        fidelity_function=fidelity,
        value=val,
        comps=traj.components[statesymb],
        goal=traj.goal[statesymb],
        statedim=traj.dims[statesymb],
        zdim=traj.dim,
        T=traj.T
    )
end



# function FinalStateFidelityConstraint(
#     val::Float64,
#     statesymb::Symbol,
#     statedim::Int;
#     fidelity_function::Function=fidelity
# )
#     return FinalFidelityConstraint(;
#         fidelity_function=fidelity_function,
#         value=val,
#         statesymb=statesymb,
#         statedim=statedim
#     )
# end

"""
    ComplexModulusContraint(<keyword arguments>)

Returns a NonlinearInequalityConstraint on the complex modulus of a complex control

# Arguments
- `R::Union{Float64,Nothing}=nothing`: the maximum allowed complex modulus
- `comps::Union{AbstractVector{Int},Nothing}=nothing`: the components of the complex control,
   both the real and imaginary parts
- `times::Union{AbstractVector{Int},Nothing}=nothing`: the times at which the constraint is applied
- `zdim::Union{Int,Nothing}=nothing`: the dimension of a single time step of the trajectory
- `T::Union{Int,Nothing}=nothing`: the number of time steps
"""
function ComplexModulusContraint(;
    R::Union{Float64, Nothing}=nothing,
    comps::Union{AbstractVector{Int}, Nothing}=nothing,
    times::Union{AbstractVector{Int}, Nothing}=nothing,
    zdim::Union{Int, Nothing}=nothing,
    T::Union{Int, Nothing}=nothing,
)
    @assert !isnothing(R) "must provide a value R, s.t. |z| <= R"
    @assert !isnothing(comps) "must provide components of the complex number"
    @assert !isnothing(times) "must provide times"
    @assert !isnothing(zdim) "must provide a z dimension"
    @assert !isnothing(T) "must provide a T"

    @assert length(comps) == 2 "component must represent a complex number and have dimension 2"

    params = Dict{Symbol, Any}()

    params[:type] = :ComplexModulusContraint
    params[:R] = R
    params[:comps] = comps
    params[:times] = times
    params[:zdim] = zdim
    params[:T] = T

    gₜ(xₜ, yₜ) = [R^2 - xₜ^2 - yₜ^2]
    ∂gₜ(xₜ, yₜ) = [-2xₜ, -2yₜ]
    μₜ∂²gₜ(μₜ) = sparse([1, 2], [1, 2], [-2μₜ, -2μₜ])

    @views function g(Z⃗)
        r = zeros(length(times))
        for (i, t) ∈ enumerate(times)
            zₜ = Z⃗[slice(t, comps, zdim)]
            xₜ = zₜ[1]
            yₜ = zₜ[2]
            r[i] = gₜ(xₜ, yₜ)[1]
        end
        return r
    end

    ∂g_structure = []

    for (i, t) ∈ enumerate(times)
        push!(∂g_structure, (i, index(t, comps[1], zdim)))
        push!(∂g_structure, (i, index(t, comps[2], zdim)))
    end

    @views function ∂g(Z⃗; ipopt=true)
        ∂ = spzeros(length(times), zdim * T)
        for (i, t) ∈ enumerate(times)
            zₜ = Z⃗[slice(t, comps, zdim)]
            xₜ = zₜ[1]
            yₜ = zₜ[2]
            ∂[i, slice(t, comps, zdim)] = ∂gₜ(xₜ, yₜ)
        end
        if ipopt
            return [∂[i, j] for (i, j) in ∂g_structure]
        else
            return ∂
        end
    end

    μ∂²g_structure = []

    for t ∈ times
        push!(
            μ∂²g_structure,
            (
                index(t, comps[1], zdim),
                index(t, comps[1], zdim)
            )
        )
        push!(
            μ∂²g_structure,
            (
                index(t, comps[2], zdim),
                index(t, comps[2], zdim)
            )
        )
    end

    function μ∂²g(Z⃗, μ; ipopt=true)
        μ∂² = spzeros(zdim * T, zdim * T)
        for (i, t) ∈ enumerate(times)
            t_slice = slice(t, comps, zdim)
            μ∂²[t_slice, t_slice] = μₜ∂²gₜ(μ[i])
        end
        if ipopt
            return [μ∂²[i, j] for (i, j) in μ∂²g_structure]
        else
            return μ∂²
        end
    end

    return NonlinearInequalityConstraint(
        g,
        ∂g,
        ∂g_structure,
        μ∂²g,
        μ∂²g_structure,
        length(times),
        params
    )
end

"""
    ComplexModulusContraint(symb::Symbol, R::Float64, traj::NamedTrajectory)

Returns a ComplexModulusContraint for the complex control NamedTrajector symbol
where R is the maximum allowed complex modulus.
"""
function ComplexModulusContraint(
    symb::Symbol,
    R::Float64,
    traj::NamedTrajectory;
    times=1:traj.T
)
    @assert symb ∈ traj.names
    return ComplexModulusContraint(;
        R=R,
        comps=traj.components[symb],
        times=times,
        zdim=traj.dim,
        T=traj.T
    )
end


abstract type LinearConstraint <: AbstractConstraint end

"""
    constrain!(opt::Ipopt.Optimizer, vars::Vector{MOI.VariableIndex}, cons::Vector{LinearConstraint}, traj::NamedTrajectory; verbose=false)

Supplies a set of LinearConstraints to  IPOPT using MathOptInterface

"""
function constrain!(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    cons::Vector{LinearConstraint},
    traj::NamedTrajectory;
    verbose=false
)
    for con in cons
        if verbose
            println("applying constraint: ", con.label)
        end
        con(opt, vars, traj)
    end
end

"""
    trajectory_constraints(traj::NamedTrajectory)

Implements the initial and final value constraints and bounds constraints on the controls
and states as specified by traj.

"""

function trajectory_constraints(traj::NamedTrajectory)
    cons = AbstractConstraint[]

    init_names = []

    # add initial equality constraints
    for (name, val) ∈ pairs(traj.initial)
        ts = [1]
        js = traj.components[name]
        con_label = "initial value of $name"
        eq_con = EqualityConstraint(ts, js, val, traj.dim; label=con_label)
        push!(cons, eq_con)
        push!(init_names, name)
    end

    final_names = []

    # add final equality constraints
    for (name, val) ∈ pairs(traj.final)
        ts = [traj.T]
        js = traj.components[name]
        con_label = "final value of $name"
        eq_con = EqualityConstraint(ts, js, val, traj.dim; label=con_label)
        push!(cons, eq_con)
        push!(final_names, name)
    end

    # add bounds constraints
    for (name, bound) ∈ pairs(traj.bounds)
        if name ∈ init_names && name ∈ final_names
            ts = 2:traj.T-1
        elseif name ∈ init_names && !(name ∈ final_names)
            ts = 2:traj.T
        elseif name ∈ final_names && !(name ∈ init_names)
            ts = 1:traj.T-1
        else
            ts = 1:traj.T
        end
        js = traj.components[name]
        con_label = "bounds on $name"
        bounds = collect(zip(bound[1], bound[2]))
        bounds_con = BoundsConstraint(ts, js, bounds, traj.dim; label=con_label)
        push!(cons, bounds_con)
    end

    return cons
end



"""
    struct EqualityConstraint

Represents a linear equality constraint.

# Fields
- `ts::AbstractArray{Int}`: the time steps at which the constraint is applied
- `js::AbstractArray{Int}`: the components of the trajectory at which the constraint is applied
- `vals::Vector{R}`: the values of the constraint
- `vardim::Int`: the dimension of a single time step of the trajectory
- `label::String`: a label for the constraint

"""
struct EqualityConstraint <: LinearConstraint
    ts::AbstractArray{Int}
    js::AbstractArray{Int}
    vals::Vector{R} where R
    vardim::Int
    label::String
end

function EqualityConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{R, Vector{R}},
    vardim::Int;
    label="unlabeled equality constraint"
) where R

    @assert !(isa(val, Vector{R}) && isa(j, Int))
        "if val is an array, j must be an array of integers"

    @assert isa(val, R) ||
        (isa(val, Vector{R}) && isa(j, AbstractArray{Int})) &&
        length(val) == length(j) """
    if j and val are both arrays, dimensions must match:
        length(j)   = $(length(j))
        length(val) = $(length(val))
    """

    if isa(val, R) && isa(j, AbstractArray{Int})
        val = fill(val, length(j))
    end

    return EqualityConstraint(
        [t...],
        [j...],
        [val...],
        vardim,
        label
    )
end


function (con::EqualityConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    for t in con.ts
        for (j, val) in zip(con.js, con.vals)
            MOI.add_constraints(
                opt,
                vars[index(t, j, con.vardim)],
                MOI.EqualTo(val)
            )
        end
    end
end

struct BoundsConstraint <: LinearConstraint
    ts::AbstractArray{Int}
    js::AbstractArray{Int}
    vals::Vector{Tuple{R, R}} where R <: Real
    vardim::Int
    label::String
end

function BoundsConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{Tuple{R, R}, Vector{Tuple{R, R}}},
    vardim::Int;
    label="unlabeled bounds constraint"
) where R <: Real

    @assert !(isa(val, Vector{Tuple{R, R}}) && isa(j, Int))
        "if val is an array, var must be an array of integers"

    if isa(val, Tuple{R,R}) && isa(j, AbstractArray{Int})

        val = fill(val, length(j))

    elseif isa(val, Tuple{R, R}) && isa(j, Int)

        val = [val]
        j = [j]

    end

    @assert *([v[1] <= v[2] for v in val]...) "lower bound must be less than upper bound"

    return BoundsConstraint(
        [t...],
        j,
        val,
        vardim,
        label
    )
end

function BoundsConstraint(
    t::Union{Int, AbstractArray{Int}},
    j::Union{Int, AbstractArray{Int}},
    val::Union{R, Vector{R}},
    vardim::Int;
    label="unlabeled bounds constraint"
) where R <: Real

    @assert !(isa(val, Vector{R}) && isa(j, Int))
        "if val is an array, var must be an array of integers"

    if isa(val, R) && isa(j, AbstractArray{Int})

        bounds = (-abs(val), abs(val))
        val = fill(bounds, length(j))

    elseif isa(val, R) && isa(j, Int)

        bounds = (-abs(val), abs(val))
        val = [bounds]
        j = [j]

    elseif isa(val, Vector{R})

        val = [(-abs(v), abs(v)) for v in val]

    end

    return BoundsConstraint(
        [t...],
        j,
        val,
        vardim,
        label
    )
end

function (con::BoundsConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    for t in con.ts
        for (j, (lb, ub)) in zip(con.js, con.vals)
            MOI.add_constraints(
                opt,
                vars[index(t, j, con.vardim)],
                MOI.GreaterThan(lb)
            )
            MOI.add_constraints(
                opt,
                vars[index(t, j, con.vardim)],
                MOI.LessThan(ub)
            )
        end
    end
end

struct TimeStepBoundsConstraint <: LinearConstraint
    bounds::Tuple{R, R} where R <: Real
    Δt_indices::AbstractVector{Int}
    label::String

    function TimeStepBoundsConstraint(
        bounds::Tuple{R, R} where R <: Real,
        Δt_indices::AbstractVector{Int},
        T::Int;
        label="time step bounds constraint"
    )
        @assert bounds[1] < bounds[2] "lower bound must be less than upper bound"
        return new(bounds, Δt_indices, label)
    end
end

function (con::TimeStepBoundsConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    for i ∈ con.Δt_indices
        MOI.add_constraints(
            opt,
            vars[i],
            MOI.GreaterThan(con.bounds[1])
        )
        MOI.add_constraints(
            opt,
            vars[i],
            MOI.LessThan(con.bounds[2])
        )
    end
end

struct TimeStepEqualityConstraint <: LinearConstraint
    val::R where R <: Real
    Δt_indices::AbstractVector{Int}
    label::String

    function TimeStepEqualityConstraint(
        val::R where R <: Real,
        Δt_indices::AbstractVector{Int};
        label="unlabeled time step equality constraint"
    )
        return new(val, Δt_indices, label)
    end
end

function (con::TimeStepEqualityConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    for i ∈ con.Δt_indices
        MOI.add_constraints(
            opt,
            vars[i],
            MOI.EqualTo(con.val)
        )
    end
end

struct TimeStepsAllEqualConstraint <: LinearConstraint
    Δt_indices::AbstractVector{Int}
    label::String

    function TimeStepsAllEqualConstraint(
        Δt_indices::AbstractVector{Int};
        label="time step all equal constraint"
    )
        return new(Δt_indices, label)
    end

    function TimeStepsAllEqualConstraint(
        Δt_symb::Symbol,
        traj::NamedTrajectory;
        label="time step all equal constraint"
    )
        Δt_comp = traj.components[Δt_symb][1]
        Δt_indices = [index(t, Δt_comp, traj.dim) for t = 1:traj.T]
        return new(Δt_indices, label)
    end
end

function (con::TimeStepsAllEqualConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    N = length(con.Δt_indices)
    for i = 1:N-1
        Δtᵢ = MOI.ScalarAffineTerm(1.0, vars[con.Δt_indices[i]])
        minusΔt̄ = MOI.ScalarAffineTerm(-1.0, vars[con.Δt_indices[end]])
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction([Δtᵢ, minusΔt̄], 0.0),
            MOI.EqualTo(0.0)
        )
    end
end

struct L1SlackConstraint <: LinearConstraint
    var_name::Symbol
    slack_names::Vector{Symbol}
    indices::AbstractVector{Int}
    times::AbstractVector{Int}
    label::String

    function L1SlackConstraint(
        name::Symbol,
        traj::NamedTrajectory;
        indices=1:traj.dims[name],
        times=(name ∈ keys(traj.initial) ? 2 : 1):traj.T,
        label="L1 slack constraint on $name"
    )
        @assert all(i ∈ 1:traj.dims[name] for i ∈ indices)
        s1_name = Symbol("s1_$name")
        s2_name = Symbol("s2_$name")
        slack_names = [s1_name, s2_name]
        add_component!(traj, s1_name, rand(length(indices), traj.T))
        add_component!(traj, s2_name, rand(length(indices), traj.T))
        return new(name, slack_names, indices, times, label)
    end
end

function (con::L1SlackConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    for t ∈ con.times
        for (s1, s2, x) in zip(
            slice(t, traj.components[con.slack_names[1]], traj.dim),
            slice(t, traj.components[con.slack_names[2]], traj.dim),
            slice(t, traj.components[con.var_name][con.indices], traj.dim)
        )
            MOI.add_constraints(
                opt,
                vars[s1],
                MOI.GreaterThan(0.0)
            )
            MOI.add_constraints(
                opt,
                vars[s2],
                MOI.GreaterThan(0.0)
            )
            t1 = MOI.ScalarAffineTerm(1.0, vars[s1])
            t2 = MOI.ScalarAffineTerm(-1.0, vars[s2])
            t3 = MOI.ScalarAffineTerm(-1.0, vars[x])
            MOI.add_constraints(
                opt,
                MOI.ScalarAffineFunction([t1, t2, t3], 0.0),
                MOI.EqualTo(0.0)
            )
        end
    end
end

end
