```@meta
CurrentModule = QuantumCollocation
```

# QuantumCollocation.jl

*Direct Collocation for Quantum Optimal Control* ([arXiv](https://arxiv.org/abs/2305.03261))

## Motivation

In quantum optimal control, we are interested in finding a pulse sequence $a_{1:T-1}$ to drive a quantum system and realize a target gate $U_{\text{goal}}$. We formulate this problem as a nonlinear program (NLP) of the form

```math
\begin{aligned}
\underset{U_{1:T}, a_{1:T-1}, \Delta t_{1:T-1}}{\text{minimize}} & \quad \ell(U_T, U_{\text{goal}})\\
\text{ subject to } & \quad f(U_{t+1}, U_t, a_t, \Delta t_t) = 0 \\
\end{aligned}
```

where $f$ defines the dynamics, implicitly, as constraints on the states and controls, $U_{1:T}$ and $a_{1:T-1}$, which are both free variables in the solver. This optimization framework is called *direct collocation*.  For details of our implementation please see our award-winning paper [Direct Collocation for Quantum Optimal Control](https://arxiv.org/abs/2305.03261).

The gist of the method is that the dynamics are given by the solution to the Schrodinger equation, which results in unitary evolution given by $\exp(-i \Delta t H(a_t))$, where $H(a_t)$ is the Hamiltonian of the system and $\Delta t$ is the timestep.  We can approximate this evolution using Pade approximants:

```math
\begin{aligned}
f(U_{t+1}, U_t, a_t, \Delta t_t) &= U_{t+1} - \exp(-i \Delta t_t H(a_t)) U_t \\
&\approx U_{t+1} - B^{-1}(a_t, \Delta t_t) F(a_t, \Delta t_t) U_t \\
&= B(a_t, \Delta t_t) U_{t+1} - F(a_t, \Delta t_t) U_t \\
\end{aligned}
```

where $B(a_t)$ and $F(a_t)$ are the *backward* and *forward* Pade operators and are just polynomials in $H(a_t)$. 

This implementation is possible because direct collocation allows for the dynamics to be implicit. Since numerically calculating matrix exponentials inherently requires an approximation -- the Padé approximant is commonly used -- utilizing this formulation significantly improves performance, as, at least here, no matrix inversion is required.


## Index

```@raw html
<div class="mermaid">
graph TD
    subgraph QuantumCollocations.jl
        A["<code>QuantumCollocations.jl</code>"]
       
        subgraph Integrators
            B["<code>UnitaryPadeIntegrator</code>"]
        end
       
        subgraph QuantumSystems
            C["<code>G</code>"]
            D["<code>AbstractQuantumSystem</code>"]
            E["<code>QuantumSystem</code>"]
            F["<code>QuantumSystemCoupling</code>"]
            G["<code>H</code>"]
        end
       
        subgraph ProblemTemplate
            H["<code>UnitaryMinimumTimeProblem</code>"]
            I["<code>UnitarySmoothPulseProblem</code>"]
        end
       
        subgraph QuantumUtils
            J["<code>annihilate</code>"]
            K["<code>create</code>"]
            L["<code>kron_from_dict</code>"]
            M["<code>operator_from_dict</code>"]
            N["<code>quad</code>"]
            O["<code>number</code>"]
            P["<code>quantum_state</code>"]
            Q["<code>vec^-1</code>"]
        end
       
        subgraph Losses
            S["<code>unitary_fidelity</code>"]
            R["<code>isovec_unitary_fidelity</code>"]
        end
    end

    A --> Integrators
    A --> QuantumSystems
    A --> ProblemTemplate
    A --> QuantumUtils
    A --> Losses
</div>
```

```@index
```

