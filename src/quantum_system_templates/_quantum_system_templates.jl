module QuantumSystemTemplates

export TransmonSystem
export TransmonDipoleCoupling
export MultiTransmonSystem

using ..QuantumUtils
using ..QuantumSystems

using LinearAlgebra
using SparseArrays

include("transmons.jl")

end
