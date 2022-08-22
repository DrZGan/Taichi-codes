import taichi as ti

ti.init(arch=ti.cpu) #initialization arch ti.cpu/ti.gpu
ti.init(default_ip=ti.i64)
ti.init(default_fp=ti.f64) #change default types to 64 bits
#default signed int: ti.i 32
#default floating point: ti.f32


#python-scope 
def foo1():
    print("This is a normal python function")
    
#taichi-scope   (python-scope 
#                       - ti.kernel (only can be called in python-scope) 
#                           - ti.func (only can be called in ti.kernel)
@ti.kernel
def foo2():
    print("This is now a taichi kernel")
    a = 1.7 #floating point
    b = ti.cast(a, ti.i32) # change type to i32
    c = ti.cast(b, ti.f32) # change type to f32
    d = ti.cast(c, ti.f64) # change type to f64
    print("b = ", b) # b = 1
    print("c = ", c) # c = 1.0
    print("d = ", d)

    #predefined keywords for compound types:
    x = ti.Vector([1.0,0.0,0.0])
    print(x)
    print(x[0])
    y = ti.Matrix([[1.5,1.2],[1.3,1.4]])
    print(y)
    print(y[0,0])
    z = ti.Struct(v1 = x, v2 = y, l = 1)
    print(z.v1)



foo1()
foo2()