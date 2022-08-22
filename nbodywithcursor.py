import taichi as ti
import numpy as np

ti.init(arch=ti.cpu) #initialization arch ti.cpu/ti.gpu
G = 1 # gravitataional constant is 6.67408e-11, using 1 here for simplicity
PI = 3.1415926
res = 512
# global control
paused = ti.field(ti.i32, ()) # a scalar i32


N = 500 # number of planets
m = 5 # nuit mass
galaxy_size = 0.4 # galaxy size
planet_radius = 2 # planet radius for rendering
init_vel = 120 # inital veclocity

# a big planet
M = 2000
big_planet_radius = 15

# hale
#start = ti.field(ti.i32, shape = ())
#ishale = ti.field(ti.i32, shape = N)

h = 1e-5 # time stpe
substepping = 10 # the number of sub-iterations within a time step

#declare fields (pos, vel, force of the planets)
# 2d problem
pos = ti.Vector.field(2, ti.f32, N)
vel = ti.Vector.field(2, ti.f32, N)
force = ti.Vector.field(2, ti.f32, N)
Ishaled = ti.field(ti.i32, N)

big_pos = ti.Vector.field(2, ti.f32, 1)
big_pos[0] = [0.5, 0.5] # initial position of the big planet

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
                r = diff.norm(1e-3) #norm of Vector diff and minimum value is 1e-5 (clamp to 1e-5)

                f = - G * m**2 * diff / r**3 # gravitational force f is a vector

                force[i] += f # gravitational force on planet i

    # force due to the big planet
    for i in range(N):
        diff = pos[i] - big_pos[0]

        if diff.norm() < 3e-2:
            Ishaled[i] = 1
        else:
            Ishaled[i] = 0

        r = diff.norm(1e-3)
        f = - G * m * M * diff / r**3
        force[i] += f

@ti.kernel  
def update():  # update each planet's vel and pos based on gravity 
    dt = h/substepping # time step 
    for i in range(N):
        
        if Ishaled[i] == 0:
            vel[i] += dt * force[i] / m
            pos[i] += dt * vel[i]
            # collision detection at edges, flip the velocity
            if pos[i][0] < 0.0 or pos[i][0] > 1.0:
                vel[i][0] *= -1
            if pos[i][1] < 0.0 or pos[i][1] > 1.0:
                vel[i][1] *= -1
        elif Ishaled[i] ==1:
            pos[i] = [-1, -1]
            vel[i] = [0, 0]

            # collisions between particles 1: brute force 2: sweep and prune 3: grid parition
            
#start the simulation
gui = ti.GUI('N-body problem', (res, res)) # create a window of resolution 512*512

'''
rgb = (0.4, 0.8, 1) # RGB color (0,0,0) = black, (1,1,1) = white
hex = ti.rgb_to_hex(rgb)
rgb2 = ti.hex_to_rgb(hex)
print(int(hex))
print(rgb2)
'''
initialize()

while gui.running: # update frames, intervel is time step h

    for e in gui.get_events(ti.GUI.PRESS): #event processing
        if e.key == ti.GUI.ESCAPE:  # 'Esc'
            exit()
        elif e.key == 'r':  # 'r'
            initialize()
        elif e.key == ti.GUI.SPACE:  # 'space'
            paused[None] = not paused[None]
       # elif e.key == ti.GUI.LMB:
        #    big_pos[0] = [e.pos[0], e.pos[1]]
    if(gui.is_pressed(ti.GUI.LMB)):
        big_pos[0] = [gui.get_cursor_pos()[0], gui.get_cursor_pos()[1]]
            
            
            
    if not paused[None]:
        for i in range(substepping): # run substepping times for each time step
            compute_force()
            update()

    gui.clear(0x112F41) # Hex code of the color: 0x000000 = black, 0xffffff = white

#    gui.circles(pos.to_numpy(), color = int( ti.rgb_to_hex((1.0,1.0,1.0)) ), radius = planet_radius)
    gui.circles(pos.to_numpy(), color = 0xffffff, radius = planet_radius)
    # relative position is ranging from (0.0, 0.0) lower left corner to (1.0, 1.0) upper right coner
    
    gui.circles(big_pos.to_numpy(), color = 0xffd700, radius = big_planet_radius)

    gui.fps_limit = 80
    gui.show()


