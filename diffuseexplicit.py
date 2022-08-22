import taichi as ti

ti.init(arch = ti.cpu, debug = True, cpu_max_num_threads = 1, advanced_optimization = True)
# make sure the debug mode is on. it is good for checking if you read out of an array range

ti.init(kernel_profiler = True)  # create a profiler telling the time each part takes
ti.clear_kernel_profile_info() #clear previous profile (use it if necessary)

paused = False
save_images = False
# problem setting
n = 128 # number of cells in each direction
scatter = 4 # pixels to visualize each cell
res = n * scatter

h = 1e-3 # time step
substep = 1
dx = 0.5 # cell size
k = 50.0 # thermal diffusivity

#heat source parameters
t_max = 300.0 # in Celsius
t_laser = ti.field(ti.f32, shape = ())   # global varialbe has to use field type
t_laser[None] = 300.0
t_min = 0.0
heat_center = ti.Vector.field(2, ti.f32, 1)
heat_center[0] = [float(n // 2)*dx, float(n // 2)*dx]
heat_radius = 2.1

laser_on = ti.field(ti.i32, shape = ())
laser_on[None] = 0
    
#temperature fields at time t and t+1
t_n =  ti.field(ti.f32, shape = (n, n))
t_np1 = ti.field(ti.f32, shape = (n, n))

#visualization
pixels = ti.Vector.field(3, ti.f32, shape = (res, res)) # RGB field 

@ti.kernel
def init():
    for i, j in t_n:
        x = float(i) * dx
        y = float(j) * dx
        '''
        if((x - heat_center[0][0])**2 + (y - heat_center[0][1])**2 <= heat_radius**2):
            t_n[i,j] = t_laser[None]
            t_np1[i,j] = t_laser[None]
        else:
            t_n[i,j] = t_min
            t_np1[i,j] = t_min
        '''
        t_n[i,j] = t_min
        t_np1[i,j] = t_min    

@ti.kernel
def update_source():
    for i, j in t_n:
        x = float(i) * dx
        y = float(j) * dx
        if((x - heat_center[0][0])**2 + (y - heat_center[0][1])**2 <= heat_radius**2):
            if(laser_on[None]):
                t_np1[i,j] = t_laser[None]



@ti.kernel
def diffuse(dt: ti.f32):
    c = dt * k / dx**2
    for i,j in t_n:
        t_np1[i, j] = t_n[i, j] # no flux boundary condition
        if i > 0:
            t_np1[i, j] += c * (t_n[i-1, j] - t_n[i, j])
        if i < n-1:
            t_np1[i, j] += c * (t_n[i+1, j] - t_n[i, j])
        if j > 0:
            t_np1[i, j] += c * (t_n[i, j-1] - t_n[i, j])
        if j < n-1:
            t_np1[i, j] += c * (t_n[i, j+1] - t_n[i, j])

       
def update_source_and_commit():
    update_source()
    t_n.copy_from(t_np1)

@ti.func
def get_color(v, vmin, vmax):
    c = ti.Vector([1.0, 1.0, 1.0]) # white

    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    dv = vmax - vmin

    if v < (vmin + 0.25 * dv):
        c[0] = 0 #Red
        c[1] = 4 * (v - vmin) / dv #Green
        c[2] = 1 #Blue
    elif v < (vmin + 0.5 * dv):
        c[0] = 0
        c[1] = 1
        c[2] = 1 + 4 * (vmin + 0.25*dv -v) / dv
    elif v < (vmin + 0.75 * dv):
        c[0] = 4 * (v - vmin - 0.5*dv) / dv
        c[1] = 1
        c[2] = 0
    else:
        c[0] = 1
        c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv
        c[2] = 0
    
    return c

@ti.kernel
def temperature_to_color(t: ti.template(), color: ti.template(), tmin: ti.f32, tmax: ti.f32):
    for i, j in t:
        for k, l in ti.ndrange(scatter, scatter):
            color[i*scatter+k,j*scatter+l] = get_color(t[i,j], tmin, tmax)

#GUI

my_gui = ti.GUI("diffuse", (res, res))

init()
i = 0

while my_gui.running:
     

    for e in my_gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.SPACE:
            paused = not paused
        elif e.key == ti.GUI.ESCAPE:  # 'Esc'
            exit()
        elif e.key == 'i':
            save_images = not save_images
            print(f"Exporting images to images\output_{i:05}.png")
        elif e.key == 'r':
            init()
            i = 0
        elif e.key == ti.GUI.LMB:
                laser_on[None] = 1

        elif e.key == ti.GUI.RMB:
                laser_on[None] = 0


    if(my_gui.is_pressed(ti.GUI.LMB)):
        heat_center[0] = [my_gui.get_cursor_pos()[0]*float(n)*dx, my_gui.get_cursor_pos()[1]*float(n)*dx]


    if not paused:
        for sub in range(substep):
            diffuse(h/substep)
            update_source_and_commit()

    temperature_to_color(t_np1, pixels, t_min, t_max)
    my_gui.set_image(pixels)
    if save_images and not paused:
        my_gui.show(f"images\output_{i:05}.png")
        i += 1
    else:
        my_gui.show()

        
        