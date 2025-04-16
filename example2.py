import numpy as np
from sympy import symbols, sin
import matplotlib.pyplot as plt

from DifferentialGeometry import RiemannGeometry





if __name__ == '__main__':
    # Sphere, Coordinates(θ, ϕ)
    a, x, y = symbols('a, x, y', real = True)
    var = [x, y]
    g = np.diag([a**2, a**2*sin(x)**2])
    Sphere = RiemannGeometry(g, var)
    
    # solve geodesic equation
    sol = Sphere.solve_geodesic_equation(104)

    # Generate coordinates for plotting
    theta, phi = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)

    # Convert spherical coordinates to Cartesian coordinates
    θ = np.sin(phi) * np.cos(theta)
    ϕ = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sphere surface
    ax.plot_surface(θ, ϕ, z, color='black', alpha=0.1, edgecolor='black', linewidth=0.1)
    ax.scatter(0, 0, 0, color='r', marker='x')

    # Extract coordinates for the geodesic curve
    theta = sol[:,0]
    phi = sol[:,1]
    curve_x = np.sin(theta) * np.cos(phi)
    curve_y = np.sin(theta) * np.sin(phi)
    curve_z = np.cos(theta)

    # Plot the geodesic curve on the sphere
    ax.plot(curve_x, curve_y, curve_z, color='r', label='Curve on Sphere', alpha=0.8, linewidth=5)

    # Set legend and title
    ax.legend()
    ax.set_title('Sphere')

    # Display the plot
    plt.show()