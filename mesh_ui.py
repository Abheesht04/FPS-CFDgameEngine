import numpy as np
import pyvista as pv
import vtk

# ---------- Load TetGen nodes ----------
node_file = "refined_strict.node"
ele_file  = "refined_strict.ele"

# Read nodes
with open(node_file, 'r') as f:
    lines = f.readlines()

num_nodes = int(lines[0].split()[0])
nodes = []
for line in lines[1:num_nodes+1]:
    parts = line.strip().split()
    x, y, z = map(float, parts[1:4])
    nodes.append([x, y, z])
nodes = np.array(nodes)

# Read tets
with open(ele_file, 'r') as f:
    lines = f.readlines()

num_tets = int(lines[0].split()[0])
tets = []
for line in lines[1:num_tets+1]:
    parts = line.strip().split()
    tet = [int(parts[i])-1 for i in range(1,5)]  # 0-based indexing
    tets.append(tet)
tets = np.array(tets)

# ---------- Create PyVista Unstructured Grid ----------
cells = np.hstack([np.hstack([[4], tet]) for tet in tets])
celltypes = np.full(len(tets), pv.CellType.TETRA)

grid = pv.UnstructuredGrid(cells, celltypes, nodes)

# ---------- Compute cell quality via VTK ----------
quality_filter = vtk.vtkCellQuality()
quality_filter.SetInputData(grid)
quality_filter.SetQualityMeasureToScaledJacobian()  # change metric if desired
quality_filter.Update()

qgrid = pv.wrap(quality_filter.GetOutput())
grid.cell_data["Quality"] = qgrid.cell_data["CellQuality"]

print(f"Quality min={grid['Quality'].min():.4f}, "
      f"max={grid['Quality'].max():.4f}, "
      f"mean={grid['Quality'].mean():.4f}")

# ---------- Interactive Visualization ----------
plotter = pv.Plotter()
plotter.add_mesh(
    grid,
    scalars="Quality",
    show_edges=True,
    opacity=0.5,
    cmap="viridis",
)
plotter.add_scalar_bar(title="Scaled Jacobian")
plotter.add_axes()
plotter.show_bounds()
plotter.show()
