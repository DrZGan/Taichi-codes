import taichi as ti

ti.init(arch=ti.cpu) #initialization arch ti.cpu/ti.gpu
ti.init(packed = True) # no padding for the data
# if packed = False (default) shape = (18, 65) will be padded to (32, 128)

# swinging taichi pattern: a 512*512 sparse grid

y = ti.field(dtype = ti.i32) # a scalar field, each member is a i32
block1 = ti.root.pointer(ti.ij, 8) # 8*8 block1 cells
block2 = block1.pointer(ti.ij, 4) # each block1 cell has 4*4 block2 cells
block3 = block2.pointer(ti.ij, 4) # each block2 cell has 4*4 block3 cells
block3.dense(ti.ij, 4).place(y) # each block3 cell has 4*4 dense pixels which are i32

# dense version: y = ti.field(dtype = ti.i32, shape = (512, 512))
# taichi uses ti.SparseMatrixBuilder to solve sparse matrices

# reading an inactive voxel returns 0

# a example of creation and manipulation of a sparse grid:

use_bitmask = False

x = ti.field(dtype = ti.i32)
level1 = ti.root.pointer(ti.ij, (4,4))
if use_bitmask == True:
    pixel = level1.bitmasked(ti.ij, (2,2))
else:
    pixel = level1.dense(ti.ij, (2,2))

pixel.place(x)
#x is a 8*8 scalar field: 4*2 * 4*2

x[2,3] = 2
x[5,6] = 3

@ti.kernel
def sparse_struct_for():
    for i,j in x:
        print('field x[{}, {}] = {}'.format(i, j, x[i,j]))

    for i,j in level1:
        print('Active block: [{}, {}]'.format(i, j))

print('use_bitmask = {}'.format(use_bitmask))

sparse_struct_for()

# if use_bitmask == True (.bitmasked), the for loop only goes through x[2,3] and x[5,6]
# if use_bitmask == False (.dense), the SNode (Structured node) also activates other pixels in block [1,1] when x[2,3] 
# is activated.
#  
# explicitly manipulating and querying sparsity

z = ti.field(dtype = ti.i32)

lv1 = ti.root.pointer(ti.ij, (4,4))  #lv1 (4,4)
lv2 = lv1.pointer(ti.ij, (2,2))      #lv2 (8,8)
pix = lv2.dense(ti.ij, (2,2)).place(z) #z (16,16)

@ti.kernel
def sparse_api_demo():
    ti.activate(lv1, [0,1]) # activate a block
    ti.activate(lv2, [1,2])
    #ti.activate(z, [10,10]) # this is not supported

    for i,j in z:
        print('field x[{}, {}] = {}'.format(i, j, z[i, j]))
    # outputs:
    #field x[2, 4] = 0
    #field x[2, 5] = 0
    #field x[3, 4] = 0
    #field x[3, 5] = 0   

    for i, j in lv1:
        print('active lv1: [{}, {}]'.format(i,j))
    # output [0, 1]
    for i, j in lv2:
        print('active lv2: [{}, {}]'.format(i,j))
    # output [1, 2]
    for j in range(4):
        print('activity of level 2 [1, {}] = {}'.format(j, ti.is_active(lv2, [1, j])))

    #ouputs:
    #activity of level 2 [1, 0] = 0
    #activity of level 2 [1, 1] = 0
    #activity of level 2 [1, 2] = 1
    #activity of level 2 [1, 3] = 0    

    ti.deactivate(lv2, [1,2])  # deactivate a block
    for i, j in lv2:
        print('active lv2: [{}, {}]'.format(i,j))
    # nothing

    print(ti.rescale_index(z,lv1,[0,15])) #compute the ancestor index given a descendant index
    print(ti.rescale_index(z,lv2,[0,15]))

    ti.activate(lv2, [1,2])
print('******test explicit manipulation of sparsity***')

sparse_api_demo()

@ti.kernel 
def check_activity(snode: ti.template(), i: ti.i32, j:ti.i32):
    print(ti.is_active(snode,[i,j]))



check_activity(lv2, 1, 2)  # output 1

lv2.deactivate_all()   # deactivate all nodes in lv2

check_activity(lv2, 1, 2)  # output 0
check_activity(lv1, 0, 1)  # output 1

ti.deactivate_all_snodes()
check_activity(lv1, 0, 1)  # output 0

#when deactivation happens, the taichi runtime automatically recycles and zero-fills memory of the deactivated containers

a= 11 // 2  # the floor division // rounds the result down to the nearest integer
print(a) # a = 5 