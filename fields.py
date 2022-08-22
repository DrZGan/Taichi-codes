from textwrap import fill
import taichi as ti

ti.init(arch=ti.cpu) #initialization arch ti.cpu/ti.gpu

# a field is a global N-dimensional array of elements
# global: can be read/written from both taichi- and python- scopes
# N-d: Scalar N = 0, Vector N = 1, Matrix N = 2, Tensor N = 3, 4, 5,...
# elements can be scalar, vector, matrix, struct.

# Scalar fields 
width, height = 640, 480
# Declaration
energy  = ti.field(dtype = ti.f32, shape = ())  # 0D: a scalar
linear_array = ti.field(dtype = ti.i32, shape = 128) # 1D scalar array

gray_scale_image = ti.field(dtype = ti.u8, shape = (width, height)) # 2D scalar matrix

volumetric_data = ti.field(ti.f32, shape = (32, 32, 32)) # 3D scalar tensor

print(linear_array.shape)  #meta data
print(volumetric_data.dtype)
# field values are initially 0 !
# fields are always accessed by indices. For 0D: x[None] instead of x

@ti.kernel
def fill_image():
    for i,j in gray_scale_image:
        gray_scale_image[i,j] = 255*ti.random() #generate a gray-scale image with random pixel values
    #ti.random() # (0,1) random number

fill_image()

gui = ti.GUI('gray-scale image with random values', (width, height))
while gui.running:
    gui.set_image(gray_scale_image)
    gui.show()

# Vector fields
n,w,h = 3,128,64
vec_field = ti.Vector.field(n=3, dtype = ti.f32, shape=(w,h))
#n = 3 means that the number of components of the vectors is 3.

@ti.kernel    # kernels should be called from python-scope, not nested kernels
def fill_vector():
    for i,j in vec_field:   # for loop at the outermost scope is parallelized (only)
                            # Break is not supported in th parallel for loop
        for k in ti.static (range(n)):  # serialized in each parallel thread
        #ti.static unrolls the inner loops
            vec_field[i,j][k] = ti.random() #two indexing operators used to access a member of a vector field

fill_vector()
print(vec_field[w-1,h-1][n-1])

#Matrix field
# the strain or stress tensor is a 3 by 3 matrix in the 3d space:
stress_field = ti.Matrix.field(n=3, m=3, dtype= ti.f32, shape= (100,100))

element = stress_field[50,50]
member = stress_field[50,50][0,0]

print(element)
print(member)

#Struct fields
# 1D field of particles with position, velocity, acceleration, and mass can be declared as:
particle_field = ti.Struct.field({
    "pos": ti.types.vector(3, ti.f32),
    "vel": ti.types.vector(3, ti.f32),
    "acc": ti.types.vector(3, ti.f32),
    "mass": ti.f32}, shape=(100))

particle_field[0] #local ti.Struct
particle_field[0].pos = ti.Vector([0.0, 0.0, 0.0])
particle_field[1].pos[0] = 1.0 # set the first position component (x-coord) of the second particle to 1.0 

particle_field.mass #global ti.Vector.field
particle_field.mass.fill(1.0) #set the mass of all particles to be 1
print(particle_field.mass) #print glocal mass field




x = ti.field(dtype = ti.i32, shape = 128) # 1D scalar array
x.fill(10)
total = 0.0
ti.kernel
def sum():
    for i in range(128):
        total += x[i]
    
print(total)