import taichi as ti
import numpy as np

ti.init(arch=ti.cpu) #initialization arch ti.cpu/ti.gpu
G = 1 # gravitataional constant is 6.67408e-11, using 1 here for simplicity
PI = 3.1415926


N = 300 # number of planets
m = 5 # nuit mass
galaxy_size = 0.4 # galaxy size
planet_radius = 2 # planet radius for rendering
init_vel = 120 # inital veclocity

h = 1e-5 # time stpe
substepping = 10 # the number of sub-iterations within a time step

#declare fields (pos, vel, force of the planets)
# 2d problem
pos = ti.Vector.field(2, ti.f32, N)
vel = ti.Vector.field(2, ti.f32, N)
force = ti.Vector.field(2, ti.f32, N)

@ti.kernel
def initialize(): #initialize pos, vel, force of each planet
    center = ti.Vector([0.5,0.5])
    for i in range(N):
        theta = ti.random() * 2 * PI  # theta = (0, 2 pi)
        r = (ti.sqrt(ti.random()) * 0.7 + 0.3) * galaxy_size # r = (0.3 1)*galaxy_size
        offset = r * ti.Vector([ti.cos(theta), ti.sin(theta)]) #
        pos[i] = center + offset
        vel[i] = [-offset.y, offset.x] # vel direction is perpendicular to its offset
        vel[i] *= init_vel

@ti.kernel
def compute_force():

    #clear force
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])
    
    #compute gravitataional force
    for i in range(N):
        for j in range(N):
            if i != j:
                diff  = pos[i] - pos[j]
                r = diff.norm(1e-5) #norm of Vector diff and minimum value is 1e-5

                f = - G * m**2 * diff / r**3 # gravitational force f is a vector

                force[i] += f # gravitational force on planet i

@ti.kernel  
def update():  # update each planet's vel and pos based on gravity 
    dt = h/substepping # time step 
    for i in range(N):
        vel[i] += dt * force[i] / m
        pos[i] += dt * vel[i]

#start the simulation
gui = ti.GUI('N-body problem', (512, 512)) # create a window of resolution 512*512

'''
rgb = (0.4, 0.8, 1) # RGB color (0,0,0) = black, (1,1,1) = white
hex = ti.rgb_to_hex(rgb)
rgb2 = ti.hex_to_rgb(hex)
print(int(hex))
print(rgb2)
'''
initialize()

while gui.running: # update frames, intervel is time step h

    for i in range(substepping): # run substepping times for each time step
        compute_force()
        update()

    gui.clear(0x112F41) # Hex code of the color: 0x000000 = black, 0xffffff = white

#    gui.circles(pos.to_numpy(), color = int( ti.rgb_to_hex((1.0,1.0,1.0)) ), radius = planet_radius)
    gui.circles(pos.to_numpy(), color = 0xffffff, radius = planet_radius)
    gui.fps_limit = 80
    gui.show()


