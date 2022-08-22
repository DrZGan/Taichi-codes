import taichi as ti

ti.init(arch = ti.cpu, debug = True, cpu_max_num_threads = 1, advanced_optimization = True)
# make sure the debug mode is on. it is good for checking if you read out of an array range

ti.init(kernel_profiler = True)  # create a profiler telling the time each part takes
ti.clear_kernel_profile_info() #clear previous profile (use it if necessary)

n = 512

x = ti.field(dtype = ti.i32)
res = n + n // 4 + n // 16 + n // 64
# n // 64 : extra pixels between block1 cells
# n // 16 : extra pixels between block2 cells
# n //  4 : extra pixels between block3 cells


img = ti.field(dtype = ti.f32, shape = (res, res))

block1 = ti.root.pointer(ti.ij, 8)
block2 = block1.pointer(ti.ij, 4)
block3 = block2.pointer(ti.ij, 4)
block3.dense(ti.ij, 4).place(x)



@ti.kernel
def activate(t: ti.f32): # define a rotating taichi field x
    for i,j in ti.ndrange(n, n):
        p = ti.Vector([i, j]) / n
        p = ti.Matrix.rotation2d(ti.sin(t)) @ (p - 0.5) + 0.5

        # rotating coordinates, @ means matrix muliplication in taichi

        if ti.taichi_logo(p) == 0:
            x[i,j] = 1

@ti.func
def scatter(i):  # jump extra pixels between blocks (they are 1: black)
    return i + i // 4 + i // 16 + i // 64 + 2


#temp = ti.field(dtype = ti.f32, shape = (5,5))
@ti.kernel
def paint():
    for i, j in ti.ndrange(n, n):
        t = x[i, j]
        block1_index = ti.rescale_index(x, block1, [i, j]) # get indice of ancestor levels
        block2_index = ti.rescale_index(x, block2, [i, j])
        block3_index = ti.rescale_index(x, block3, [i, j])

        t += ti.is_active(block1, block1_index)
        t += ti.is_active(block2, block2_index)
        t += ti.is_active(block3, block3_index)

        img[scatter(i), scatter(j)] = 1 - t / 4
        # frame locations: img = 0.05 (black)
        # locations where block1 are inactive, img = 1 (white)
        # locations where block1 are active but block2 are inactive, img = 0.75
        # block1 and block2 are active but block3 are inactive, img = 0.5
        # block1-3 are active, img = 0.25

img.fill(0.05)
gui = ti.GUI('sparse grids',(res,res))

for i in range(100):
    block1.deactivate_all() #deactivate all blocks
    activate(i * 0.05) # control rotation speed
    paint()
    gui.set_image(img)
    gui.show()

   
#    for i, j in temp: # this is equalivent to for i, j in ndrange(5,5)
#        print('temp[{}][{}]={}'.format(i,j,temp[i,j]) )
#    for i, j in ti.ndrange((1,4+1), (1,4+1)): # GOOD!
#        print('temp[{}][{}]={}'.format(i,j,temp[i,j]) )
paint()

ti.print_kernel_profile_info('count')