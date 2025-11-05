import numpy as np

# ----------------------------
# TetGen Readers/Writers
# ----------------------------
def load_node(filename):
    with open(filename) as f:
        lines = f.readlines()
    n = int(lines[0].split()[0])
    return np.array([list(map(float, l.split()[1:4])) for l in lines[1:n+1]])

def load_ele(filename):
    with open(filename) as f:
        lines = f.readlines()
    n = int(lines[0].split()[0])
    return np.array([list(map(int, l.split()[1:5])) for l in lines[1:n+1]], dtype=np.int32)

def write_node(filename, nodes):
    with open(filename, "w") as f:
        f.write(f"{len(nodes)} 3 0 0\n")
        for i, p in enumerate(nodes, start=1):
            f.write(f"{i} {p[0]} {p[1]} {p[2]}\n")

def write_ele(filename, tets):
    with open(filename, "w") as f:
        f.write(f"{len(tets)} 4 0\n")
        for i, t in enumerate(tets, start=1):
            f.write(f"{i} {t[0]} {t[1]} {t[2]} {t[3]}\n")

# ----------------------------
# Quality Metric
# ----------------------------
def tet_quality(nodes, tets):
    a = nodes[tets[:,0]-1]
    b = nodes[tets[:,1]-1]
    c = nodes[tets[:,2]-1]
    d = nodes[tets[:,3]-1]
    edges = np.stack([
        np.linalg.norm(a-b, axis=1),
        np.linalg.norm(a-c, axis=1),
        np.linalg.norm(a-d, axis=1),
        np.linalg.norm(b-c, axis=1),
        np.linalg.norm(b-d, axis=1),
        np.linalg.norm(c-d, axis=1)
    ], axis=1)
    lmax = edges.max(axis=1)
    vol = np.abs(np.einsum('ij,ij->i', b-a, np.cross(c-a, d-a))) / 6.0
    q = np.zeros_like(vol)
    mask = lmax > 1e-12
    q[mask] = (6 * np.cbrt(vol[mask])) / lmax[mask]
    return q

# ----------------------------
# Strict adaptive refinement
# ----------------------------
def strict_refine(nodes, tets, min_quality=0.25, z_detect_ratio=0.1, max_iter=8):
    nodes = nodes.tolist()
    tets = tets.tolist()

    for it in range(max_iter):
        q = tet_quality(np.array(nodes), np.array(tets))
        centroids = np.array([np.mean([nodes[i-1] for i in tet], axis=0) for tet in tets])

        # --- Automatic detection of worst region ---
        z_sorted = np.sort(centroids[:,2])
        z_cut = np.percentile(z_sorted, z_detect_ratio * 100)  # lowest 10% region by default
        bad_idx = [i for i, qi in enumerate(q) if qi < min_quality and centroids[i,2] <= z_cut]

        print(f"Iteration {it+1}: refining {len(bad_idx)} bad tets in z<{z_cut:.2f} region")

        if not bad_idx:
            break

        new_tets = []
        bad_nodes = set()
        for i, tet in enumerate(tets):
            if i in bad_idx:
                pts = np.array([nodes[j-1] for j in tet])
                centroid = pts.mean(axis=0)
                nodes.append(centroid.tolist())
                ci = len(nodes)
                bad_nodes.update(tet)
                new_tets += [
                    [tet[0], tet[1], tet[2], ci],
                    [tet[0], tet[1], tet[3], ci],
                    [tet[0], tet[2], tet[3], ci],
                    [tet[1], tet[2], tet[3], ci],
                ]
            else:
                new_tets.append(tet)
        tets = new_tets

        # --- Local smoothing on bad node region ---
        nodes_np = np.array(nodes)
        for nid in bad_nodes:
            connected = []
            for tet in tets:
                if nid in tet:
                    connected += [n for n in tet if n != nid]
            if connected:
                avg_pos = np.mean(nodes_np[np.array(connected)-1], axis=0)
                nodes_np[nid-1] = 0.8 * nodes_np[nid-1] + 0.2 * avg_pos  # weighted relaxation
        nodes = nodes_np.tolist()

        print(f"  total nodes={len(nodes)}, total tets={len(tets)}")

    return np.array(nodes), np.array(tets)

# ----------------------------
# Main
# ----------------------------
nodes = load_node("new_mesh_fixed.1.node")
tets = load_ele("new_mesh_fixed.1.ele")

nodes_r, tets_r = strict_refine(nodes, tets, min_quality=0.40, z_detect_ratio=0.1, max_iter=6)

write_node("refined_strict.node", nodes_r)
write_ele("refined_strict.ele", tets_r)

print("✅ Strict adaptive nose refinement done.")
