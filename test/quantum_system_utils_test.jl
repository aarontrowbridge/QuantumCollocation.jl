"""
Tests: QuantumSystemUtils submodule
"""


@testitem "Lie algebra basis" begin
    H_ops = Dict(
        "X" => GATES[:X],
        "Y" => GATES[:Y],
        "Z" => GATES[:Z]
    )

    # Check 1 qubit with complete basis
    gen = map(A -> kron_from_dict(A, H_ops), ["X", "Y"])
    basis = operator_algebra(gen, return_layers=false, verbose=false)
    @test length(basis) == size(first(gen), 1)^2-1

    # Check 1 qubit with complete basis and layers
    gen = map(A -> kron_from_dict(A, H_ops), ["X", "Y"])
    basis, layers = operator_algebra(gen, return_layers=true, verbose=false)
    @test length(basis) == size(first(gen), 1)^2-1

    # Check 1 qubit with subspace
    gen = map(A -> kron_from_dict(A, H_ops), ["X"])
    basis = operator_algebra(gen, verbose=false)
    @test length(basis) == 1

    # Check 2 qubit with complete basis
    gen = map(
        AB -> kron_from_dict(AB, H_ops),
        ["XX+YY","XI", "YI", "IY", "IX"]
    )
    basis = operator_algebra(gen, verbose=false)
    @test length(basis) == size(first(gen), 1)^2-1

    # Check 2 qubit with linearly dependent basis
    gen = map(
        AB -> kron_from_dict(AB, H_ops),
        ["XX+YY", "XI", "XI", "IY", "IX"]
    )
    basis = operator_algebra(gen, verbose=false)
    @test length(basis) == length(gen)

    # Check 2 qubit with pair of 1-qubit subspaces
    gen = map(
        AB -> kron_from_dict(AB, H_ops),
         ["XI", "YI", "IY", "IX"]
    )
    basis = operator_algebra(gen, verbose=false)
    @test length(basis) == 2 * (2^2 - 1)
end


@testitem "Lie Algebra reachability" begin
    using LinearAlgebra

    H_ops = Dict(
        "X" => GATES[:X],
        "Y" => GATES[:Y],
        "Z" => GATES[:Z]
    )

    # Check 1 qubit with complete basis
    gen = map(A -> kron_from_dict(A, H_ops), ["X", "Y"])
    target = H_ops["Z"]
    @test is_reachable(target, gen, compute_basis=true, verbose=false)

    # System
    sys = QuantumSystem([GATES[:X], GATES[:Y], GATES[:Z]])
    target = GATES[:Z]
    @test is_reachable(target, sys)

    # System with drift
    sys = QuantumSystem(GATES[:Z], [GATES[:X]])
    target = GATES[:Z]
    @test is_reachable(target, sys)

    # Check 2 qubit with complete basis
    XI = GATES[:X] ⊗ GATES[:I]
    IX = GATES[:I] ⊗ GATES[:X]
    YI = GATES[:Y] ⊗ GATES[:I]
    IY = GATES[:I] ⊗ GATES[:Y]
    XX = GATES[:X] ⊗ GATES[:X]
    YY = GATES[:Y] ⊗ GATES[:Y]
    ZI = GATES[:Z] ⊗ GATES[:I]
    IZ = GATES[:I] ⊗ GATES[:Z]
    ZZ = GATES[:Z] ⊗ GATES[:Z]

    complete_gen = [XX+YY, XI, YI, IX, IY]
    incomplete_gen = [XI, ZZ]
    r = [0, 1, 2, 3, 4]
    r /= norm(r)
    R2 = exp(-im * sum([θ * H for (H, θ) in zip(complete_gen, r)]))
    CZ = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]
    CX = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
    
    # Pass
    @test is_reachable(R2, complete_gen)
    @test is_reachable(CZ, complete_gen)
    @test is_reachable(CX, complete_gen)
    @test is_reachable(XI, complete_gen)

    # Mostly fail
    @test !is_reachable(R2, incomplete_gen)
    @test !is_reachable(CZ, incomplete_gen)
    @test !is_reachable(CX, incomplete_gen)
    @test is_reachable(XI, incomplete_gen)

    # QuantumSystems
    complete_gen_sys = QuantumSystem(complete_gen)
    incomplete_gen_sys = QuantumSystem(incomplete_gen)
    # Pass
    @test is_reachable(R2, complete_gen_sys)
    @test is_reachable(CZ, complete_gen_sys)
    @test is_reachable(CX, complete_gen_sys)
    @test is_reachable(XI, complete_gen_sys)

    # Mostly fail
    @test !is_reachable(R2, incomplete_gen_sys)
    @test !is_reachable(CZ, incomplete_gen_sys)
    @test !is_reachable(CX, incomplete_gen_sys)
    @test is_reachable(XI, incomplete_gen_sys)
end

@testitem "Lie Algebra subspace reachability" begin
    # TODO: implement tests
end
