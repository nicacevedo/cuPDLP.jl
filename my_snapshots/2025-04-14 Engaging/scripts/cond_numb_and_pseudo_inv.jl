# For small matrices
using LinearAlgebra

# Define a matrix A
A = [1.0 2.0; 3.0 4.0; 5.0 6.0]

# Compute pseudoinverse
A_dagger = pinv(A)
println("Pseudoinverse of A:")
println(A_dagger)

# Compute singular values
singular_values = svdvals(A)
println("Singular values of A:")
println(singular_values)

# Extract max and min singular values
sigma_max = maximum(singular_values)
sigma_min = minimum(singular_values[singular_values .> 0])
println("Max singular value (σ_max): $sigma_max")
println("Min singular value (σ_min^+): $sigma_min")

# Compute condition number
condition_number = sigma_max / sigma_min
println("Condition number of A: $condition_number")


# For large sparse matrices
using SparseArrays
using LinearAlgebra
using Arpack

# Define a large sparse matrix
n = 10_000
A = sprandn(n, n ÷ 2, 0.01)  # Sparse matrix (10,000 x 5,000) with 1% density
println("Sparse matrix A created with size $(size(A))")

# Compute largest singular value (σ_max)
σ_max = svds(A; nev=1, which=:LM).values[1]
println("Largest singular value (σ_max): $σ_max")

# Compute smallest singular value (σ_min^+)
σ_min = svds(A; nev=1, which=:SM).values[1]
println("Smallest singular value (σ_min^+): $σ_min")

# Compute condition number
condition_number = σ_max / σ_min
println("Condition number of A: $condition_number")

# (Optional) Compute pseudoinverse (not practical for very large matrices)
A_dagger = pinv(Matrix(A))  # Convert to dense, use only for smaller matrices
println("Pseudoinverse (if feasible):")
println(A_dagger)
