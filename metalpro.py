import taichi as ti

ti.init(arch=ti.cpu) #initialization arch ti.cpu/ti.gpu

@ti.kernel
def copy(x:ti.template(), y:ti.template()):
    for I in ti.grouped(y):
        x[I] = y [I]

    # I is a vector with dimensionality same to y
    # if y is 0D, I = ti.Vector([])
    # if y is 1D, I = ti.Vector([i])
    # if y is 2D, I = ti.Vector([i, j])
    # if y is 3D, I = ti.Vector([i, j, k])


@ti.kernel
def foo():
    for i in ti.static(range(4)):
        print(i) # is equivalent to print (0), 1, 2, 3. No parallelization

# ti. static
#   1. ti.static(varible) to tell complier it is a constant
#   2. ti.static(range(4)) to serialize a for loop
#   3. force loop unrolling for element index access (indices into compound taichi types must be complie-time constant)
x = ti.Vector.field (n = 3, dtype = ti.f32, shape=(8))
@ti.kernel
def reset():
    for i in x:
        for j in ti.static(range(x.n)):
            x[i][j] = 0
            # the inner loop must be unrolled since j is an index for accessing a vector
    