using CUDA
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using CUDA.CUSPARSE





# C = CUDA.CUSPARSE.CuSparseMatrixCSR(C)




function Sparse_MM(D1,A,D2)
    A = D1 * A * D2
    return A
end

function Sparse_MM2(D1,A,D2)
    CUDA.CUSPARSE.gemm!(
             'N',   # 1st matrix is not transpose
             'N',   # 2nd matrix is not transpose
             1,     # 1*A*B
             D1, # A
             A,     # B
             0,                            
             A,     # C
             'O',   # Secret
        # CUDA.CUSPARSE.CUSPARSE_SPMM_CSR_ALG2, # determinstic algorithm(?)
        ) 
    CUDA.CUSPARSE.gemm!(
             'N',   # 1st matrix is not transpose
             'N',   # 2nd matrix is not transpose
             1,     # 1*A*B
             A, # A
             D2,     # B
             0,                            
             A,     # C
             'O',   # Secret
        ) 
    return A
end

function main()
    n=1000
    m=100
    d=0.1
    sample_size = 100000
    normal_time = 0
    sparse_time = 0
    for i in 1:sample_size
        println(repeat("-", 100))
        D1 = sprand(m,m,d)
        A = sprand(m,n,d)
        D2 = sprand(n,n,d) 
        D1 = CUDA.CUSPARSE.CuSparseMatrixCSR(D1)
        A = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
        D2 = CUDA.CUSPARSE.CuSparseMatrixCSR(D2)
   
        t0 = time()
        Sparse_MM(D1,A,D2)
        t1 = time()

        println("Normal one: ", t1-t0)
        normal_time += t1-t0

        t0 = time()
        Sparse_MM2(D1,A,D2)
        t1 = time()

        println("Sparse one: ", t1-t0)
        sparse_time += t1-t0

    end
    println("Final times:")
    println(normal_time)
    println(sparse_time)
end

main()
# CUDA.CUSPARSE.gemm!(
#          'N',   # 1st matrix is not transpose
#          'N',   # 2nd matrix is not transpose
#          1,     # 1*A*B
#          D1, # A
#          A,     # B
#          0,                            
#          C,     # C
#          'O',   # Secret
#     )

# A[rand(axes(A,2)),rand(axes(A,2))] = 0
# A = sparse(A)
# println(sparse(A))


# CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2

