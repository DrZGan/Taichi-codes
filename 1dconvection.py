import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
ti.init(arch = ti.cpu, debug = True, packed = True, kernel_profiler = True, default_fp=ti.f64)
#ti.init(arch = ti.cpu, debug = True, cpu_max_num_threads = 1, advanced_optimization = True)
# make sure the debug mode is on. it is good for checking if you read out of an array range
#ti.init(packed = True) # no padding for the data
# if packed = False (default) shape = (18, 65) will be padded to (32, 128)
#ti.init(kernel_profiler = True)  # create a profiler telling the time each part takes
ti.clear_kernel_profile_info() #clear previous profile (use it if necessary)

n = 50 # number of grid nodes
L = 1.0 # length of the geometry (m)
rho = 1.0 # density (kg/m^3)
u = 12 # velocity (m/s)
Tao = 0.1 # diffusion coefficient (kg/m/s)
phi0 = 1 # boundary condition at x = 0
phiL = 0 # boundary condition at x = L

dx = L / float(n) # mesh size
dt = 1e-4 # pseudo time step
interations = 30000 # pseudo iterations
scheme = 'QUICK'   # 'Central differencing', 'Upwind', 'QUICK'

# Quadratic Upstream Interpolation for Convective Kinetics (QUICK)

# start solving the convection-diffusion equation

phi_n =  ti.field(ti.f64, shape = n)
phi_np1 = ti.field(ti.f64, shape = n)
error_field = ti.field(ti.f64, shape = n)
# f32 can represent 1e-38, f64 can represent 1e-308, USE f64!!
@ti.kernel 
def init_BC():
    phi_n[0] = 1
    phi_n[n-1] = 0 # this assignment is not necessary

    phi_np1[0] = 1
    phi_np1[n-1] = 0 # this assignment is not necessary    

@ti.kernel
def convect_diffuse():
    ap0 = rho * dx / dt
    aw = aw()
    ae = ae()
    ap = ap()
    aww = aww()
    aee = aee()
    for i in range(1, n-1): # 1 2 ... n-2 [range(n) = 0, 1, 2, ..., n-1]

        if scheme == 'Upwind' or scheme == 'Central differencing':
            phi_np1[i] = ((ap0 - ap) * phi_n[i] + aw * phi_n[i-1] + ae * phi_n[i+1]) / ap0
        
        elif scheme == 'QUICK': # second ordered upwind scheme
            if i == 1: # left boundary
                phi_np1[i] = ((ap0 - ap) * phi_n[i] + aw * phi_n[i-1] + ae * phi_n[i+1] + aww * phi_n[0] + aee * phi_n[i+2]) / ap0
            elif i == n-2: # right boundary
                phi_np1[i] = ((ap0 - ap) * phi_n[i] + aw * phi_n[i-1] + ae * phi_n[i+1] + aww * phi_n[i-2] + aee * phi_n[n-1]) / ap0 
            else:
                phi_np1[i] = ((ap0 - ap) * phi_n[i] + aw * phi_n[i-1] + ae * phi_n[i+1] + aww * phi_n[i-2] + aee * phi_n[i+2]) / ap0

        else:
            print('Error in computing convect_diffuse()')   

def commit():
    phi_n.copy_from(phi_np1)     

@ti.kernel
def error_function(): # degree of approaching steady-state 1e-6
    ap0 = rho * dx / dt
    aw = aw()
    aww = aww()
    ae = ae()
    aee = aee()
    ap = ap()

    if scheme == 'Upwind' or scheme == 'Central differencing':
        for i in range(1, n-1):
            residual = aw * phi_np1[i-1] + ae * phi_np1[i+1] - ap * phi_np1[i]
            error_field[i] = residual / (ap0 * phi_np1[i])
            #error_field[i] =  ( phi_np1[i])
    
    elif scheme == 'QUICK':
        for i in range(2, n-2):
            residual = aw * phi_np1[i-1] + ae * phi_np1[i+1] + aww * phi_np1[i-2] + aee * phi_np1[i+2] - ap * phi_np1[i]
            error_field[i] = residual / (ap0 * phi_np1[i])
    
    else:
            print('Error in computing error_function()')   

@ti.func
def aw():  # ti.func does not need to be type-hinted for the return value
    aw = 0.0 # ti.func does not support multiple returns, so we use a local varible to store the results
    Dw = Tao / dx
    Fw = rho * u
    Fe = rho * u
    alpha_w = 0.0
    alpha_e = 0.0        

    if rho*u > 0:
        alpha_w = 1.0
        alpha_e = 1.0

    if scheme == 'Central differencing':
        aw = Dw + Fw / 2.0
    elif scheme == 'Upwind':
        aw = Dw + max(Fw, 0)
    elif scheme == 'QUICK':
        aw = Dw + 6.0/8.0*alpha_w*Fw + 1.0/8.0*alpha_e*Fe + 3.0/8.0*(1-alpha_w)*Fw
    else:
        print('Error in computing aw')
    #print(aw)
    return aw 

@ti.func
def aww():  # for QUICK scheme
    
    Fw = rho * u   
    alpha_w = 0.0          

    if rho*u > 0:
        alpha_w = 1.0
    #print(-1.0/8.0*alpha_w*Fw)    
    return -1.0/8.0*alpha_w*Fw


@ti.func
def ae():  # ti.func does not need to be type-hinted for the return value
    ae = 0.0 # ti.func does not support multiple returns, so we use a local varible to store the results
    De = Tao / dx
    Fw = rho * u
    Fe = rho * u
    alpha_w = 0.0
    alpha_e = 0.0        

    if rho*u > 0:
        alpha_w = 1.0
        alpha_e = 1.0

    if scheme == 'Central differencing':
        ae = De - Fe / 2.0
    elif scheme == 'Upwind':
        ae = De + max(0, -Fe)
    elif scheme == 'QUICK':
        ae = De - 3.0/8.0*alpha_e*Fe - 6.0/8.0*(1-alpha_e)*Fe - 1.0/8.0*(1-alpha_w)*Fw
    
    else:
        print('Error in computing ae')
    #print(ae)
    return ae 

@ti.func
def aee():  # for QUICK scheme
    
    Fe = rho * u   
    alpha_e = 0.0          

    if rho*u > 0:
        alpha_e = 1.0
    #print(1.0/8.0*(1-alpha_e)*Fe)  
    return 1.0/8.0*(1-alpha_e)*Fe


@ti.func
def ap():  # ti.func does not need to be type-hinted for the return value
    
    ap = 0.0
    Fw = rho * u
    Fe = rho * u
    if scheme == 'Upwind' or scheme == 'Central differencing':
        ap = aw() + ae() + (Fe - Fw)

    elif scheme == 'QUICK':
        ap = aw() + ae() + aww() + aee() + (Fe - Fw)

    else:
        print('Error in computing ap')        
    return ap


@ti.kernel
def convergence_criteria(): 

    print('Limit for explicit scheme (ap0-ap>0) = {}'.format(rho * dx / dt - ap()))
    print('Peclet number (F/D=rho*u/(Tao/dx)<2 for central differencing scheme) = {}'.format(rho*u/(Tao/dx)))
    print('Courant number (u*dt/dx<1) = {}'.format(u*dt/dx))    


# ---- start the interation ---------
init_BC()

for i in range(interations):
    convect_diffuse()
    commit()
    error_function()
print(error_field)
print('Residual error (degree of approaching steady-state) = {}'.format(np.linalg.norm(error_field.to_numpy())))
# check convergence criteria

print('Check convergence criteria:')
convergence_criteria()

#ti.print_kernel_profile_info('count')    
#print(phi_n[50]) #Taichi will check memory overflow

# ----Complete the interation -------

# plot analytical solution and numerical solution
x = np.linspace(0, L, n)
y_ana = (np.exp(rho * u * x / Tao) - 1) / (np.exp(rho * u * L / Tao) - 1) * (phiL - phi0) + phi0
y_num = phi_np1.to_numpy()
#print(y_ana)
#print(y_num)
#plot y(x) 
#plt.rcParams["font.family"] = "Arial"
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
#ax.plot(t, x, '-o', color='blue', label='x(t)')
ax.plot(x, y_ana, 'o', color='blue', label='Analytical')
ax.plot(x, y_num, '-', color='red', label='Numerical')
ax.set_xlabel("x",fontsize=20)
ax.set_ylabel("phi",fontsize=20)
ax.tick_params(labelsize=20)
ax.legend(fontsize=20)
plt.show()
