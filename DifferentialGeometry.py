# Importing necessary libraries
import numpy as np
from sympy import Matrix, diff, simplify, symbols, Function, lambdify
from scipy.integrate import odeint




# Defining the DifferentialGeometry class
class DifferentialGeometry:
    def __init__(self, metric, variables):
        self.metric = Matrix(metric)
        self.inverse_metric = self.metric.inv()
        self.det = self.metric.det()
        self.ndim = len(metric)
        self.variables = variables
        self.parameterized_vars = [Function(var)(symbols('s')) for var in variables]



# Defining the ClassicalDifferentialGeometry class
class ClassicalDifferentialGeometry(DifferentialGeometry):
    def __init__(self, metric, variables):
        super().__init__(metric, variables)
        
        assert self.ndim == 2, "Classical differential geometry is only defined for 2D surfaces"
        self.gaussian_curvature = self.compute_gaussian_curvature()

    def compute_gaussian_curvature(self):
        det = self.det
        g = self.metric
        x = self.variables
        
        first_term = (2 * diff(diff(g[0,1], x[0]), x[1]) - diff(diff(g[0,0], x[1]), x[1]) - diff(diff(g[1,1], x[0]), x[0])) / (2 * det)
        second_term = (diff(g[0,0], x[0]) * (2 * diff(g[0,1], x[1]) - diff(g[1,1], x[0])) - (diff(g[0,0], x[1]) ** 2)) * g[1,1] / (4 * det ** 2)
        third_term = (diff(g[0,0], x[0]) * diff(g[1,1], x[1]) - 2 * diff(g[0,0], x[1]) * diff(g[1,1], x[0]) + (2 * diff(g[0,1], x[0]) - diff(g[0,0], x[1])) * (2 * diff(g[0,1], x[1]) - diff(g[1,1], x[0]))) * g[0,1] / (4 * det ** 2)
        forth_term = (diff(g[1,1], x[1]) * (2 * diff(g[0,1], x[0]) - diff(g[0,0], x[1])) - diff(g[1,1], x[0]) ** 2) * g[0,0] / (4 * det ** 2)
        
        return simplify(first_term - second_term + third_term - forth_term)



# Defining the RiemannGeometry class
class RiemannGeometry(DifferentialGeometry):
    def __init__(self, metric, variables):
        super().__init__(metric, variables)
        
        self.christoffel_symbols = self.compute_christoffel_symbols()
        self.riemann_curvature_tensor = self.compute_riemann_curvature_tensor()
        self.ricci_tensor = self.compute_ricci_tensor()
        self.ricci_scalar = self.compute_ricci_scalar()
        self.einstein_tensor = self.compute_einstein_tensor()

    def compute_christoffel_symbols(self):
        ndim = self.ndim
        g = self.metric
        inv_g = self.inverse_metric
        x = self.variables
        
        christoffel_symbols = np.zeros((ndim, ndim, ndim), dtype=object)
        
        for rho in range(ndim):
            for lamda in range(ndim):
                for nu in range(ndim):
                    for mu in range(ndim):
                        christoffel_symbols[rho,lamda,nu] += inv_g[rho,mu] * (diff(g[mu,nu], x[lamda]) + diff(g[mu,lamda], x[nu]) - diff(g[nu,lamda], x[mu])) / 2
                    
                    christoffel_symbols[rho,lamda,nu] = simplify(christoffel_symbols[rho,lamda,nu])
        
        return christoffel_symbols
    
    def compute_riemann_curvature_tensor(self):
        ndim = self.ndim    
        gamma = self.christoffel_symbols
        x = self.variables
        
        riemann_curvature_tensor = np.zeros((ndim, ndim, ndim, ndim), dtype=object)
        
        for rho in range(ndim):
            for lamda in range(ndim):
                for mu in range(ndim):
                    for nu in range(ndim):
                        for sigma in range(ndim):
                            riemann_curvature_tensor[rho,lamda,mu,nu] += (- gamma[sigma,lamda,mu] * gamma[rho,sigma,nu] + gamma[sigma,lamda,nu] * gamma[rho,sigma,mu])
                            
                        riemann_curvature_tensor[rho,lamda,mu,nu] += -diff(gamma[rho,lamda,mu], x[nu]) + diff(gamma[rho,lamda,nu], x[mu])
                        
                        riemann_curvature_tensor[rho,lamda,mu,nu] = simplify(riemann_curvature_tensor[rho,lamda,mu,nu])
        
        return riemann_curvature_tensor
    
    def compute_ricci_tensor(self):
        ndim = self.ndim
        riemann = self.riemann_curvature_tensor
        
        ricci_tensor = np.zeros((ndim, ndim), dtype=object)
        
        for mu in range(ndim):
            for nu in range(ndim):
                for lamda in range(ndim):
                    ricci_tensor[mu,nu] += riemann[lamda,mu,lamda,nu]
                    
                ricci_tensor[mu,nu] = simplify(ricci_tensor[mu,nu])
        
        return ricci_tensor
    
    def compute_ricci_scalar(self):
        ndim = self.ndim
        inv_g = self.inverse_metric
        ricci = self.ricci_tensor
        
        ricci_scalar = 0
        
        for mu in range(ndim):
            for nu in range(ndim):
                ricci_scalar += inv_g[mu,nu] * ricci[mu,nu]
        
        ricci_scalar = simplify(ricci_scalar)
        
        return ricci_scalar
    
    def compute_einstein_tensor(self):
        ndim = self.ndim
        g = self.metric
        ricci = self.ricci_tensor
        ricci_scalar = self.ricci_scalar
        
        einstein_tensor = np.zeros((ndim, ndim), dtype=object)
        
        for mu in range(ndim):
            for nu in range(ndim):
                einstein_tensor[mu,nu] = ricci[mu,nu] - (1/2) * ricci_scalar * g[mu,nu]
        
        einstein_tensor = simplify(einstein_tensor)
        
        return einstein_tensor
    
    def compute_geodesic_equation(self):
        ndim = self.ndim
        gamma = self.christoffel_symbols
        parameterized_vars = self.parameterized_vars
        
        geodesic_equation = {}
        s = symbols('s')
        
        for i in range(ndim):
            geodesic_equation[parameterized_vars[i]] = 0
            
            for j in range(ndim):
                for k in range(ndim):
                    geodesic_equation[parameterized_vars[i]] -=  gamma[i,j,k] * parameterized_vars[j].diff(s) * parameterized_vars[k].diff(s)
        
        return geodesic_equation
    
    def solve_geodesic_equation(self, seed):
        np.random.seed(seed)
        
        ndim = self.ndim
        geodesic_equation = self.compute_geodesic_equation()
        variables = self.variables
        parameterized_vars = self.parameterized_vars
        
        s = symbols('s')
        substitutes = {var.diff(s): symbols(f'{str(var)}_dot') for var in parameterized_vars}
        
        geodesic_function = {var: lambdify((*variables, *substitutes.values()), geodesic_equation[var].subs(substitutes)) for var in parameterized_vars}
        
        
        def func(y, s):
            dydt1 = [y[i+ndim] for i in range(ndim)]
            dydt2 = [geodesic_function[var](*y) for var in parameterized_vars]
            
            return dydt1 + dydt2
        
        y0 = [np.random.random() for _ in range(2*ndim)]
        t = np.linspace(0, 10, 1000)
        sol = odeint(func, y0, t)
        
        return sol