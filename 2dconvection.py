import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
ti.init(arch = ti.cpu, debug = True, packed = True, kernel_profiler = True, default_fp=ti.f64\
    ,cpu_max_num_threads = 20)

# make sure the debug mode is on. it is good for checking if you read out of an array range
#ti.init(packed = True) # no padding for the data
# if packed = False (default) shape = (18, 65) will be padded to (32, 128)
#ti.init(kernel_profiler = True)  # create a profiler telling the time each part takes
ti.clear_kernel_profile_info() #clear previous profile (use it if necessary)

nx = 100 # number of grid nodes
ny = nx
Lx = 1.0 # length of the geometry (m)
Ly = Lx
rho = 1.0 # density (kg/m^3)
ux = 3 # velocity (m/s)
uy = ux
Tao = 0.000001 # diffusion coefficient (kg/m/s)
phi_up = 100 # boundary condition at x = 0
phi_left = 0 # boundary condition at y = 0

dx = Lx / float(nx) # mesh size
dy = Ly / float(ny)

dt = 1e-4 # pseudo time step
interations = 10000 # pseudo iterations
scheme = 'QUICK'   # 'Central differencing', 'Upwind', 'QUICK'

# Quadratic Upstream Interpolation for Convective Kinetics (QUICK)

# start solving the convection-diffusion equation

phi_n =  ti.field(ti.f64, shape = (nx, ny))
phi_np1 = ti.field(ti.f64, shape = (nx, ny))
error_field = ti.field(ti.f64, shape = (nx, ny))
init_phi = ti.field(ti.f64, shape = (nx, ny))
# f32 can represent 1e-38, f64 can represent 1e-308, USE f64!!
@ti.kernel 
def init_BC():

    for j in range(ny):
        for i in range(nx):
            if j  > i:
                phi_n[i,j] = phi_up
                phi_np1[i,j] = phi_up
            else:
                phi_n[i,j] = phi_left
                phi_np1[i,j] = phi_left

    for j in range(ny): #up boundary
        phi_n[0,j] = phi_up
        phi_np1[0,j] = phi_up
        phi_n[1,j] = phi_up
        phi_np1[1,j] = phi_up    

    for i in range(nx): # left boundary
        phi_n[i,0] = phi_left
        phi_np1[i,0] = phi_left       
        phi_n[i,1] = phi_left
        phi_np1[i,1] = phi_left      

@ti.kernel
def convect_diffuse():
    ap0 = rho * dx * dy/ dt
    Su = 0.0
    aw = aw()
    ae = ae()
    an = an()
    a_s = a_s()
    ap = ap()
    aww = aww()
    aee = aee()
    ann = ann()
    ass = ass()
    for i in range(2, nx-2): # 2 ... n-3 [range(n) = 0, 1, 2, ..., n-1]
        for j in range(2, ny-2): # 2 ... n-3 [range(n) = 0, 1, 2, ..., n-1]

            if scheme == 'Upwind' or scheme == 'Central differencing':
                phi_np1[i,j] = ((ap0 - ap) * phi_n[i,j] + aw * phi_n[i-1,j] + ae * phi_n[i+1,j] +\
                    an * phi_n[i,j+1] + a_s * phi_n[i,j-1] + Su ) / ap0
            
            elif scheme == 'QUICK': # second ordered upwind scheme

                phi_np1[i,j] = ((ap0 - ap) * phi_n[i,j] + aw * phi_n[i-1,j] + ae * phi_n[i+1,j] +\
                    an * phi_n[i,j+1] + a_s * phi_n[i,j-1] + Su +\
                    aww * phi_n[i-2,j] + aee * phi_n[i+2,j] + ass * phi_n[i,j-2] + ann * phi_n[i,j+2]) / ap0
            
            else:
                print('Error in computing convect_diffuse()')   

def commit():
    phi_n.copy_from(phi_np1)     

@ti.kernel
def error_function(): # degree of approaching steady-state 1e-6
    ap0 = rho * dx * dy / dt
    aw = aw()
    aww = aww()
    ae = ae()
    aee = aee()
    ap = ap()
    an = an()
    ann = ann()
    a_s = a_s()
    ass = ass()

    if scheme == 'Upwind' or scheme == 'Central differencing':
        for i in range(2, nx-2):
            for j in range(2, ny-2):
                residual = aw * phi_np1[i-1,j] + ae * phi_np1[i+1,j] + \
                     an * phi_np1[i,j+1] + a_s * phi_np1[i,j-1] - ap * phi_np1[i,j]
                error_field[i,j] = residual / (ap0 * phi_np1[i,j])
   
    elif scheme == 'QUICK':
        for i in range(2, nx-2):
            for j in range(2, ny-2):
                residual = aw * phi_np1[i-1,j] + ae * phi_np1[i+1,j] + \
                           an * phi_np1[i,j+1] + a_s * phi_np1[i,j-1] + \
                           aww * phi_n[i-2,j] + aee * phi_n[i+2,j] + ass * phi_n[i,j-2] + ann * phi_n[i,j+2] - \
                           ap * phi_np1[i,j]
                error_field[i,j] = residual / (ap0 * phi_np1[i,j])
 

    #     for i in range(2, n-2):
    #         residual = aw * phi_np1[i-1] + ae * phi_np1[i+1] + aww * phi_np1[i-2] + aee * phi_np1[i+2] - ap * phi_np1[i]
    #         error_field[i] = residual / (ap0 * phi_np1[i])
    
    else:
            print('Error in computing error_function()')   

@ti.func
def aw():  # ti.func does not need to be type-hinted for the return value
    aw = 0.0 # ti.func does not support multiple returns, so we use a local varible to store the results
    Dw = Tao * dy / dx
    Fw = rho * ux * dy
    Fe = rho * ux * dy
    alpha_w = 0.0
    alpha_e = 0.0        

    if rho*ux*dy > 0:
        alpha_w = 1.0
        alpha_e = 1.0

    if scheme == 'Central differencing':
        aw = Dw + Fw / 2.0
    elif scheme == 'Upwind':
        aw = Dw + max(Fw, 0)
    elif scheme == 'QUICK':
        aw = Dw + 6.0/8.0*alpha_w*Fw + 1.0/8.0*alpha_e*Fe + 3.0/8.0*(1.0-alpha_w)*Fw
    else:
        print('Error in computing aw')
    #print('aw=',aw)
    return aw 

@ti.func
def aww():  # for QUICK scheme
    
    Fw = rho * ux *dy   
    alpha_w = 0.0          

    if rho*ux > 0:
        alpha_w = 1.0
    #print('aww=',-1.0/8.0*alpha_w*Fw)    
    return -1.0/8.0*alpha_w*Fw


@ti.func
def ae():  # ti.func does not need to be type-hinted for the return value
    ae = 0.0 # ti.func does not support multiple returns, so we use a local varible to store the results
    De = Tao * dy / dx
    Fw = rho * ux * dy
    Fe = rho * ux * dy
    alpha_w = 0.0
    alpha_e = 0.0        

    if rho*ux > 0:
        alpha_w = 1.0
        alpha_e = 1.0

    if scheme == 'Central differencing':
        ae = De - Fe / 2.0
    elif scheme == 'Upwind':
        ae = De + max(0, -Fe)
    elif scheme == 'QUICK':
        ae = De - 3.0/8.0*alpha_e*Fe - 6.0/8.0*(1-alpha_e)*Fe - 1.0/8.0*(1.0-alpha_w)*Fw
    
    else:
        print('Error in computing ae')
    #print('ae=',ae)
    return ae 

@ti.func
def aee():  # for QUICK scheme
    
    Fe = rho * ux * dy   
    alpha_e = 0.0          

    if rho*ux*dy > 0:
        alpha_e = 1.0
    #print('aee=',1.0/8.0*(1.0-alpha_e)*Fe)  
    return 1.0/8.0*(1.0-alpha_e)*Fe


@ti.func
def a_s():  # ti.func does not need to be type-hinted for the return value
    a_s = 0.0 # ti.func does not support multiple returns, so we use a local varible to store the results
    Ds = Tao * dx / dy
    Fs = rho * uy * dx
    Fn = rho * uy * dx
    alpha_s = 0.0
    alpha_n = 0.0        

    if rho*uy*dx > 0:
        alpha_s = 1.0
        alpha_n = 1.0

    if scheme == 'Central differencing':
        a_s = Ds + Fs / 2.0
    elif scheme == 'Upwind':
        a_s = Ds + max(0, Fs)
    elif scheme == 'QUICK':
        a_s = Ds + 6.0/8.0*alpha_s*Fs + 1.0/8.0*alpha_n*Fn + 3.0/8.0*(1.0-alpha_s)*Fs
    
    else:
        print('Error in computing as')
    #print('as=',a_s)
    return a_s 

@ti.func
def ass():  # for QUICK scheme
    
    Fs = rho * uy * dx  
    alpha_s = 0.0          

    if rho*uy*dx > 0:
        alpha_s = 1.0
    #print('ass=',-1.0/8.0*alpha_s*Fs)    
    return -1.0/8.0*alpha_s*Fs


@ti.func
def an():  # ti.func does not need to be type-hinted for the return value
    an = 0.0 # ti.func does not support multiple returns, so we use a local varible to store the results
    Dn = Tao * dx / dy
    Fs = rho * uy * dx
    Fn = rho * uy * dx
    alpha_s = 0.0
    alpha_n = 0.0        

    if rho*uy*dx > 0:
        alpha_s = 1.0
        alpha_n = 1.0

    if scheme == 'Central differencing':
        an = Dn - Fn / 2.0
    elif scheme == 'Upwind':
        an = Dn + max(0, -Fn)
    elif scheme == 'QUICK':
        an = Dn - 3.0/8.0*alpha_n*Fn - 6.0/8.0*(1.0-alpha_n)*Fn - 1.0/8.0*(1.0-alpha_s)*Fs
    
    else:
        print('Error in computing an')
    #print('an=',an)
    return an 

@ti.func
def ann():  # for QUICK scheme
    
    Fn = rho * uy * dx  
    alpha_n = 0.0          

    if rho*uy*dx > 0:
        alpha_n = 1.0
    #print('ann=',1.0/8.0*(1.0-alpha_n)*Fn)  
    return 1.0/8.0*(1.0-alpha_n)*Fn


@ti.func
def ap():  # ti.func does not need to be type-hinted for the return value
    
    ap = 0.0
    Sp = 0.0
    Fw = rho * ux * dy
    Fe = rho * ux * dy
    Fn = rho * uy * dx
    Fs = rho * uy * dx
    if scheme == 'Upwind' or scheme == 'Central differencing':
        ap = aw() + ae() + a_s() + an() - Sp + (Fe - Fw + Fn - Fs)  

    elif scheme == 'QUICK':
        ap = aw() + ae() + a_s() + an() - Sp + aww() + aee() + ass() + ann() + (Fe - Fw + Fn - Fs)

    else:
        print('Error in computing ap')    
    #print('ap=',ap)    
    return ap


@ti.kernel
def convergence_criteria(): 

    print('Limit for explicit scheme (ap0-ap>0) = {}'.format(rho * dx * dy / dt - ap()))
    print('Peclet number (F/D=rho*u/(Tao/dx)<2 for central differencing scheme) = {}'.format(rho*ux/(Tao/dx)))
    print('Courant number (u*dt/dx<1) = {}'.format(ux*dt/dx))    


# ---- start the interation ---------
init_BC()
init_phi.copy_from(phi_n)  

for i in range(interations):
    convect_diffuse()
    commit()
    error_function()
#print(error_field)
print('Residual error (degree of approaching steady-state) = {}'.format(np.linalg.norm(error_field.to_numpy())))
# check convergence criteria

print('Check stability criteria:')
convergence_criteria()

#ti.print_kernel_profile_info('count')    
#print(phi_n[50]) #Taichi will check memory overflow

# ----Complete the interation -------

# plot analytical solution and numerical solution
x = np.linspace(0, Lx, nx)
#y_ana = (np.exp(rho * u * x / Tao) - 1) / (np.exp(rho * u * L / Tao) - 1) * (phiL - phi0) + phi0
#print(phi_np1)
y_num = phi_np1.to_numpy()
y_init = init_phi.to_numpy()
y_num1d= np.zeros(nx)
y_init1d= np.zeros(nx)
for i in range(nx):
    for j in range(ny):
        if i + j == nx - 1:
            y_num1d[i]= y_num[i,j]
            y_init1d[i]= y_init[i,j]
 

#print(y_ana)
#print(y_num)
#plot y(x) 
#plt.rcParams["font.family"] = "Arial"
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
#ax.plot(t, x, '-o', color='blue', label='x(t)')
ax.plot(x, y_num1d, 'o', color='blue')
ax.plot(x, y_num1d, '--', color='red', label='Numerical')
ax.plot(x, y_init1d, '-', color='black', label='Inital')
ax.set_xlabel("A-A",fontsize=20)
ax.set_ylabel("$\phi$",fontsize=20)
ax.tick_params(labelsize=20)
ax.legend(fontsize=20)
plt.show()
