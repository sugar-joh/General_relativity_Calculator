import numpy as np
from sympy import symbols, sin, exp, Function
from DifferentialGeometry import RiemannGeometry, ClassicalDifferentialGeometry





if __name__ == '__main__':
    # Sphere, Coordinates(θ, ϕ)
    a, x, y = symbols('a, x, y', real = True)
    var = [x, y]
    g = np.diag([a**2, a**2*sin(x)**2])
    Sphere = ClassicalDifferentialGeometry(g, var)
    print(Sphere.gaussian_curvature)

    # Gauss-Bolyai-Lobachevsky plane, Klein Coordinates(x1, x2)
    a, x1, x2 = symbols('a, x1, x2', real = True)
    var = [x1, x2]
    g = np.array([[a**2*(1-x2**2)/(1-x1**2-x2**2)**2, a**2*x1*x2/(1-x1**2-x2**2)**2], 
                [a**2*x1*x2/(1-x1**2-x2**2)**2, a**2*(1-x1**2)/(1-x1**2-x2**2)**2]])
    GBL = ClassicalDifferentialGeometry(g, var)
    print(GBL.gaussian_curvature)
    
    # Saddle surface, Coordinates(x, y)
    θ, ϕ = symbols('x, y', real = True)
    var = [θ, ϕ]
    g = np.array([[1+ϕ**2, θ*ϕ], [θ*ϕ, 1+θ**2]])
    Saddle = ClassicalDifferentialGeometry(g, var)
    print(Saddle.gaussian_curvature)
    
    # Rindler spacetime, Coordinates(t, θ)
    a, t, θ = symbols('a, t, θ', real = True)
    var = [t, θ]
    g = np.diag([-exp(2*a*θ), exp(2*a*θ)])
    Rindler = RiemannGeometry(g, var)
    print(Rindler.riemann_curvature_tensor)
    
    # flat FLRW spacetime, Coordinates(t, r, θ, ϕ)
    t, r, θ, ϕ = symbols('t r θ ϕ', real=True)
    a = Function('a')(t)
    var = [t, r, θ, ϕ]
    g = np.diag([-1, a**2, a**2*r**2, a**2*r**2*sin(θ)**2])
    flat_FLRW = RiemannGeometry(g, var)
    print(flat_FLRW.einstein_tensor)