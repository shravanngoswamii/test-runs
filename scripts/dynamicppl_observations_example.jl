using DynamicPPL, Distributions

struct SymbolChoice <: DiscreteUnivariateDistribution end
Base.eltype(::Type{SymbolChoice}) = Symbol
Distributions.loglikelihood(::SymbolChoice, x::Symbol) =
    x in (:left, :right) ? -log(2) : -Inf

@model function choice(y)
    x ~ Normal()
    y ~ SymbolChoice()
end

check_model(choice(:right))
