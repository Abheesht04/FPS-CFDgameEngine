import numpy as np
import pyvista as pv

# ------------------- Load TetGen Mesh -------------------
node_file = "new_mesh_fixed.1.node"
ele_file  = "new_mesh_fixed.1.ele"

# Read nodes
with open(node_file) as f:
    lines = f.readlines()
num_nodes = int(lines[0].split()[0])
nodes = np.array([[float(x) for x in line.strip().split()[1:4]] for line in lines[1:num_nodes+1]])

# Read tetrahedra
with open(ele_file) as f:
    lines = f.readlines()
num_tets = int(lines[0].split()[0])
tets = np.array([[int(x)-1 for x in line.strip().split()[1:5]] for line in lines[1:num_tets+1]])

# Create PyVista UnstructuredGrid
cells = np.hstack([np.hstack([[4], tet]) for tet in tets])
celltypes = np.full(len(tets), pv.CellType.TETRA)
grid = pv.UnstructuredGrid(cells, celltypes, nodes)
cell_centers = grid.cell_centers().points
num_cells = len(tets)

# ------------------- CFD Initialization -------------------
U = np.zeros((num_cells, 3))   # velocity
P = np.zeros(num_cells)        # pressure
rho = 1.0
mu  = 0.001
U[:, 2] = 1.0  # inlet along +Z

# ------------------- Compute Tetra Volumes -------------------
def tet_volume(p):
    a, b, c, d = p
    return abs(np.dot(b-a, np.cross(c-a, d-a))) / 6.0

volumes = np.array([tet_volume(nodes[t]) for t in tets])

# ------------------- Naive Solver -------------------
dt = 0.001
num_steps = 100

for step in range(num_steps):
    U_new = U.copy()
    # Diffusion (simplified)
    for i, tet in enumerate(tets):
        grad = np.zeros(3)
        for j in range(4):
            grad += U[i] - U_new[i]
        U_new[i] += mu * grad / volumes[i] * dt

    # Pressure (crude incompressibility)
    P[:] = np.mean(U_new, axis=1)

    # Update velocity
    U = U_new.copy()
    
    if step % 10 == 0:
        print(f"Step {step} | mean velocity magnitude: {np.mean(np.linalg.norm(U, axis=1)):.4f}")

# ------------------- Assign CFD Fields to Grid -------------------
grid.cell_data["Velocity_Mag"] = np.linalg.norm(U, axis=1)
grid.cell_data["Pressure"] = P

# ------------------- Particle Tracing -------------------
num_particles = 200
particles = np.zeros((num_particles, 3))
particles[:, 0] = np.random.uniform(grid.bounds[0], grid.bounds[1], num_particles)
particles[:, 1] = np.random.uniform(grid.bounds[2], grid.bounds[3], num_particles)
particles[:, 2] = grid.bounds[4]

dt_particle = 0.01
particle_paths = [particles.copy()]

for _ in range(50):
    cell_idx = grid.find_containing_cell(particles)
    v = np.zeros_like(particles)
    mask = cell_idx != -1
    v[mask] = U[cell_idx[mask]]
    particles += v * dt_particle
    particle_paths.append(particles.copy())

# ------------------- Visualization -------------------
plotter = pv.Plotter()
# Velocity magnitude
plotter.add_mesh(grid, scalars="Velocity_Mag", show_edges=True, opacity=0.5, cmap="viridis")
# Pressure overlay
plotter.add_mesh(grid, scalars="Pressure", show_edges=False, opacity=0.3, cmap="coolwarm")
# Add particle paths
for path in particle_paths[::5]:
    plotter.add_points(path, color="red", point_size=3)
plotter.add_axes()
plotter.show_bounds()
plotter.show()
