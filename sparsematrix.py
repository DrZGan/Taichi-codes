import taichi as ti

ti.init(arch = ti.cpu, debug = True, cpu_max_num_threads = 1, advanced_optimization = True)
# make sure the debug mode is on. it is good for checking if you read out of an array range

ti.init(kernel_profiler = True)  # create a profiler telling the time each part takes

# only support CPU, data type is f32, storage is column-major

n = 4

# step 1: create sparse matrix builder
K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets = 100) #triplets: non-zero components

# step 2: fill the builder with data
@ti.kernel
def fill(A: ti.linalg.sparse_matrix_builder()):
    for i in range(n):
        A[i,i] += 2 # only += and -= are supported for now 
    
fill(K)

print('>>>> triplets')
K.print_triplets()

# step 3: create a sparse matrix from the builder

A = K.build()
print(A)

# basic operations like + - * @ and transpose are supported

# sparse linear solver

b = ti.field(ti.f32, shape = n)
b[0] = 1
b[n-1] = 1
print(b)

# creat a solver:
solver = ti.linalg.SparseSolver(solver_type = "LLT")
# LLT: for symmetric positive-define matrix
# LDLT: for full-ranked symmetric matrix
# LU: for square matrix
solver.analyze_pattern(A) 
solver.factorize(A)

# solve
x = solver.solve(b)

# check if the computation was successful
isSuccess = solver.info()

print(x)
print(">>>>",isSuccess)
