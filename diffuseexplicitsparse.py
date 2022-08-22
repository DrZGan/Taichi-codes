import taichi as ti
ti.init(arch = ti.cpu)
#ti.init(arch = ti.cpu, debug = True, cpu_max_num_threads = 1, advanced_optimization = True)
# make sure the debug mode is on. it is good for checking if you read out of an array range
ti.init(packed = True) # no padding for the data
# if packed = False (default) shape = (18, 65) will be padded to (32, 128)

ti.init(kernel_profiler = True)  # create a profiler telling the time each part takes
ti.clear_kernel_profile_info() #clear previous profile (use it if necessary)

paused = False
save_images = False
# problem setting
n = 200 # number of cells in each direction
scatter = 4 # pixels to visualize each cell
res = n * scatter

h = 1e-3 # time step 1 ms
substep = 1
dx = 1e-4 # cell size 0.1 mm
k = 2.5e-6 # thermal diffusivity = 10 / 5000 / 800

#heat source parameters
t_max = 2000.0 # in Kelvin
t_laser = ti.field(ti.f32, shape = ())   # global varialbe has to use field type
t_laser[None] = 2000.0
t_min = 0.0   # the minimum temperature has to be zero (0)
heat_center = ti.Vector.field(2, ti.f32, 1)
heat_center[0] = [float(n // 2)*dx, float(n // 2)*dx]
heat_radius = 4.1 * dx

laser_on = ti.field(ti.i32, shape = ())
laser_on[None] = 0  # a global flag should be a field
    
#temperature fields at time t and t+1
use_bitmask = True  # if bitmask is false, ti.is_active cannot reach the lowest level data

t_n = ti.field(dtype = ti.f32) #t_n =  ti.field(ti.f32, shape = (n, n))
t_n_lv1 = ti.root.pointer(ti.ij, (n // 2, n // 2))
if use_bitmask == True:
    t_n_pixel = t_n_lv1.bitmasked(ti.ij, (2,2))
else:
    t_n_pixel = t_n_lv1.dense(ti.ij, (2,2))
t_n_pixel.place(t_n)

t_np1 = ti.field(dtype = ti.f32) #t_np1 = ti.field(ti.f32, shape = (n, n))
t_np1_lv1 = ti.root.pointer(ti.ij, (n // 2, n // 2))
if use_bitmask == True:
    t_np1_pixel = t_np1_lv1.bitmasked(ti.ij, (2,2))
else:
    t_np1_pixel = t_np1_lv1.dense(ti.ij, (2,2))
t_np1_pixel.place(t_np1)

#visualization
pixels = ti.Vector.field(3, ti.f32, shape = (res, res)) # RGB field 

# cancel init()

@ti.kernel 
def activate(): # update computation region
    for i, j in ti.ndrange(n, n):
    # if use ti.ndrange(n, n), loop all i,j;
    # if use a field name (t_n), loop all activated cells
        x = float(i) * dx
        y = float(j) * dx

        activate_factor = 4.0
        if(laser_on[None]): # activate region around the laser
            if((x - heat_center[0][0])**2 + (y - heat_center[0][1])**2 <= (heat_radius*activate_factor)**2):
                # another way to activate:
                #   n_lv1_index = ti.rescale_index(t_n, t_n_lv1, [i, j])
                #   ti.activate(t_n_lv1, t_n_lv1_index) 
                if(not ti.is_active(t_n_pixel,[i,j])):
                    t_n[i,j] = t_min

        extended_cells = 5
        # extend the computation region due to heat transfer
        if(t_n[i,j] > t_min + 10 and t_n[i,j] < 0.01 * t_laser[None]):
            for k,l in ti.ndrange(extended_cells,extended_cells): # activate around +- cells
                if(not ti.is_active(t_n_pixel,[i+k,j+l])):
                    if(i + k <= n-1 and j + l <= n-1):
                        t_n[i+k,j+l] = t_min
                if(not ti.is_active(t_n_pixel,[i-k,j-l])):
                    if(i - k >= 0 and j - l >= 0):
                        t_n[i-k,j-l] = t_min             

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
def temperature_to_color():
    
    for i, j in ti.ndrange(n,n):
        if(ti.is_active(t_np1_pixel,[i,j])): # only show activated cells in RGB scale
            for k, l in ti.ndrange(scatter, scatter):
                pixels[i*scatter+k,j*scatter+l] = get_color(t_np1[i,j], t_min, t_max)
        else: # inactive cells show gray 
            for k, l in ti.ndrange(scatter, scatter): 
                pixels[i*scatter+k,j*scatter+l] = ti.Vector([0.4, 0.4, 0.4])            

#GUI
my_gui = ti.GUI("diffuse", (res, res))

#init()
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
            ti.deactivate_all_snodes()
            i = 0
        elif e.key == ti.GUI.LMB:
            laser_on[None] = 1

        elif e.key == ti.GUI.RMB:
            laser_on[None] = 0

    if(my_gui.is_pressed(ti.GUI.LMB)):
        heat_center[0] = [my_gui.get_cursor_pos()[0]*float(n)*dx, my_gui.get_cursor_pos()[1]*float(n)*dx]

    if not paused:
        for sub in range(substep):
            activate()
            diffuse(h/substep)
            update_source_and_commit()
    
    temperature_to_color()
    my_gui.set_image(pixels)
    if save_images and not paused:
        #my_gui.fps_limit = 50
        my_gui.show(f"images\output_{i:05}.png")
        i += 1
    else:
        #my_gui.fps_limit = 50
        my_gui.show()

#ti.print_kernel_profile_info('count')       