import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


# =============================================================================
# 1. GRID GENERATION
# =============================================================================

def generate_grid(b, d, box_edge_length):
    """
    Generates a grid of points based on primitive vectors within a specified box.
    """
    L = box_edge_length

    # Define primitive unit vectors based on geometry
    u1 = np.array([1, 0, 0])
    u2 = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3), 0])
    u3 = np.array([1 / 2, np.sqrt(3) / 6, np.sqrt(2 / 3)])

    # Scale the unit vectors
    p1, p2, p3 = (d) * u1, (d) * u2, (d) * u3
    dv = np.dot(p1, np.cross(p2, p3))

    # Transform corners to find index ranges
    M = np.array([p1, p2, p3]).T
    M_inv = np.linalg.inv(M)
    corners = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L], [L, L, 0], [L, 0, L], [0, L, L], [L, L, L]])
    corners_ijk = M_inv @ corners.T

    i_range = np.arange(int(np.floor(corners_ijk[0].min())), int(np.ceil(corners_ijk[0].max())) + 1)
    j_range = np.arange(int(np.floor(corners_ijk[1].min())), int(np.ceil(corners_ijk[1].max())) + 1)
    k_range = np.arange(int(np.floor(corners_ijk[2].min())), int(np.ceil(corners_ijk[2].max())) + 1)

    I, J, K = np.meshgrid(i_range, j_range, k_range, indexing='ij')

    XX = I * p1[0] + J * p2[0] + K * p3[0]
    YY = I * p1[1] + J * p2[1] + K * p3[1]
    ZZ = I * p1[2] + J * p2[2] + K * p3[2]

    box_mask = (XX >= 0) & (XX < L) & (YY >= 0) & (YY < L) & (ZZ >= 0) & (ZZ < L)

    print(f"Grid generated. Total points inside box: {np.sum(box_mask)}")
    return XX, YY, ZZ, box_mask, dv


# =============================================================================
# 2. SLIP SYSTEM GEOMETRY
# =============================================================================

def calculate_tetrahedron_slip_systems(b):
    """
    Calculates the slip systems based on a regular tetrahedron geometry.
    Returns: Array (4 planes, 3 directions, 2 vectors, 3 coords)
    """
    a = b
    V = np.array([
        [0, 0, 0],  # V0
        [a, 0, 0],  # V1
        [a / 2, a * np.sqrt(3) / 2, 0],  # V2
        [a / 2, a * np.sqrt(3) / 6, a * np.sqrt(2 / 3)]  # V3
    ])

    faces_indices = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    slip_systems = np.zeros((4, 3, 2, 3))

    for i, indices in enumerate(faces_indices):
        v_a, v_b, v_c = V[indices[0]], V[indices[1]], V[indices[2]]

        if i == 0:
            raw_n = np.cross(V[1] - V[0], V[2] - V[0])
        elif i == 1:
            raw_n = np.cross(V[1] - V[0], V[3] - V[0])
        elif i == 2:
            raw_n = np.cross(V[2] - V[0], V[3] - V[0])
        elif i == 3:
            raw_n = np.cross(V[2] - V[1], V[3] - V[1])

        n = raw_n / np.linalg.norm(raw_n)
        raw_directions = [v_b - v_a, v_c - v_b, v_a - v_c]

        for j, raw_b in enumerate(raw_directions):
            b_vec = raw_b / np.linalg.norm(raw_b)
            slip_systems[i, j, 0, :] = n
            slip_systems[i, j, 1, :] = b_vec

    return slip_systems


def get_fcc_slip_systems():
    """ Standard FCC slip systems (111)<110> """
    normals = [
        np.array([1, 1, 1]), np.array([-1, 1, 1]),
        np.array([1, -1, 1]), np.array([1, 1, -1])
    ]
    directions = [
        [np.array([0, 1, -1]), np.array([1, 0, -1]), np.array([1, -1, 0])],
        [np.array([0, 1, -1]), np.array([1, 0, 1]), np.array([1, 1, 0])],
        [np.array([0, 1, 1]), np.array([1, 0, -1]), np.array([1, 1, 0])],
        [np.array([0, 1, 1]), np.array([1, 0, 1]), np.array([1, -1, 0])]
    ]
    systems_array = np.zeros((4, 3, 2, 3))
    for i in range(4):
        for j in range(3):
            systems_array[i, j, 0, :] = normals[i]
            systems_array[i, j, 1, :] = directions[i][j]
    return systems_array


# =============================================================================
# 3. PLANE IDENTIFICATION & VISUALIZATION
# =============================================================================

def calculate_plane_ids(xx, yy, zz, normals):
    nx, ny, nz = xx.shape
    num_normals = len(normals)
    plane_ids_storage = np.zeros((nx, ny, nz, num_normals), dtype=np.int32)

    print("Calculating Plane IDs...")
    for idx, normal in enumerate(normals):
        dot_product = xx * normal[0] + yy * normal[1] + zz * normal[2]
        # Scale to avoid float precision issues grouping distinct planes
        dot_product_scaled = dot_product * 1e10
        dot_product_rounded = np.round(dot_product_scaled, decimals=1)
        unique_values = np.unique(dot_product_rounded)

        plane_ids_flat = np.searchsorted(unique_values, dot_product_rounded.ravel())
        plane_ids_storage[..., idx] = plane_ids_flat.reshape((nx, ny, nz))

        print(f"  Normal {idx}: Found {len(unique_values)} unique planes.")

    return plane_ids_storage


def visualize_planes(XX, YY, ZZ, box_mask, slip_plane_ids, target_normal_idx):
    ids_for_normal = slip_plane_ids[..., target_normal_idx]
    valid_ids_in_box = ids_for_normal[box_mask]

    if len(valid_ids_in_box) == 0:
        print("Error: No points found inside the box mask.")
        return

    unique_ids_in_box = np.unique(valid_ids_in_box)
    middle_index = len(unique_ids_in_box) // 2

    if middle_index + 2 < len(unique_ids_in_box):
        target_ids = [unique_ids_in_box[middle_index], unique_ids_in_box[middle_index + 1],
                      unique_ids_in_box[middle_index + 2]]
    else:
        target_ids = unique_ids_in_box[-3:]

    print(f"\nVisualizing Normal {target_normal_idx}, Planes: {target_ids}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xx_flat = XX[box_mask]
    yy_flat = YY[box_mask]
    zz_flat = ZZ[box_mask]

    ax.scatter(xx_flat[::20], yy_flat[::20], zz_flat[::20], s=1, alpha=0.05, c='gray')
    colors = ['red', 'blue', 'green']

    for i, pid in enumerate(target_ids):
        mask_plane = (valid_ids_in_box == pid)
        if np.any(mask_plane):
            ax.scatter(xx_flat[mask_plane], yy_flat[mask_plane], zz_flat[mask_plane],
                       s=50, c=colors[i % 3], label=f'Plane ID {pid}')

    ax.legend()
    plt.show()


# =============================================================================
# 4. RSS CALCULATION (UPDATED LOGIC)
# =============================================================================

def calculate_rss_and_activity(loading_dir, load_val, crss, slip_systems):
    """
    Calculates RSS and Activity.

    UPDATED LOGIC:
    1. Negative RSS is strictly inactive (0).
    2. Relative Threshold: If RSS < (Max_RSS / 5), it is inactive.
    3. CRSS Check: Must be >= CRSS.
    """

    L = np.array(loading_dir, dtype=float)
    L = L / np.linalg.norm(L)

    rss_matrix = np.zeros((4, 3))
    activity_matrix = np.zeros((4, 3), dtype=int)

    # 1. Calculate raw RSS for all systems
    for i in range(4):
        for j in range(3):
            n_vec = slip_systems[i, j, 0, :]
            b_vec = slip_systems[i, j, 1, :]
            n = n_vec / np.linalg.norm(n_vec)
            b = b_vec / np.linalg.norm(b_vec)

            schmid_factor = np.dot(L, n) * np.dot(L, b)
            rss_matrix[i, j] = load_val * schmid_factor

    # 2. Find Global Maximum Positive RSS
    # We only care about positive stress driving dislocation motion in the direction of b
    max_rss = np.max(rss_matrix)

    # If the max stress is negative or zero, nothing moves
    if max_rss <= 0:
        print("  ! Max RSS is non-positive. No slip systems active.")
        return rss_matrix, activity_matrix

    relative_threshold = max_rss / 5.0
    print(f"  > Max RSS: {max_rss:.2f} MPa. Relative Threshold (1/5): {relative_threshold:.2f} MPa")

    # 3. Determine Activity based on strict rules
    for i in range(4):
        for j in range(3):
            rss = rss_matrix[i, j]

            # Rule A: Must be positive
            if rss <= 0:
                activity_matrix[i, j] = 0
                continue

            # Rule B: Must be significant compared to the max stress
            if rss < relative_threshold:
                activity_matrix[i, j] = 0
                continue

            # Rule C: Must exceed critical resolved shear stress
            if rss >= crss:
                activity_matrix[i, j] = 1
            else:
                activity_matrix[i, j] = 0

    return rss_matrix, activity_matrix


# =============================================================================
# 5. MESH RECONSTRUCTION
# =============================================================================

def structured_mesh_from_fcc_plane(points, tolerance=1e-15):
    """ Reconstructs a structured grid from points. """
    num_points = len(points)
    if num_points < 5: return None, None, None, None, None

    # Determine Basis
    subset = points[:min(50, num_points)]
    dists = cdist(subset, subset)
    valid_dists = dists[dists > 1e-8]
    if len(valid_dists) == 0: return None, None, None, None, None
    nn_dist = np.min(valid_dists)

    a1, a2, found_basis = None, None, False

    # Try to find 120 degree neighbors
    for i in range(min(10, num_points)):
        p0 = points[i]
        dist_from_p0 = np.linalg.norm(points - p0, axis=1)
        neighbor_indices = np.where((dist_from_p0 > nn_dist * 0.95) & (dist_from_p0 < nn_dist * 1.05))[0]
        if len(neighbor_indices) < 2: continue

        vecs = points[neighbor_indices] - p0
        unit_vecs = vecs / np.linalg.norm(vecs, axis=1)[:, None]

        for j in range(len(unit_vecs)):
            for k in range(j + 1, len(unit_vecs)):
                dot = np.dot(unit_vecs[j], unit_vecs[k])
                if -0.6 < dot < -0.4:  # ~120 deg
                    a1, a2 = vecs[j], vecs[k]
                    found_basis = True;
                    break
            if found_basis: break
        if found_basis: break

    if not found_basis:  # Fallback to 60 deg
        for i in range(min(10, num_points)):
            p0 = points[i]
            dist_from_p0 = np.linalg.norm(points - p0, axis=1)
            neighbor_indices = np.where((dist_from_p0 > nn_dist * 0.95) & (dist_from_p0 < nn_dist * 1.05))[0]
            if len(neighbor_indices) < 2: continue
            vecs = points[neighbor_indices] - p0
            unit_vecs = vecs / np.linalg.norm(vecs, axis=1)[:, None]
            for j in range(len(unit_vecs)):
                for k in range(j + 1, len(unit_vecs)):
                    if 0.4 < np.dot(unit_vecs[j], unit_vecs[k]) < 0.6:
                        a1, a2 = vecs[j], vecs[k] - vecs[j]
                        found_basis = True;
                        break
                if found_basis: break
            if found_basis: break

    if not found_basis: return None, None, None, None, None

    # Map to Grid
    M = np.column_stack((a1, a2))
    origin = points[0]
    P_rel = (points - origin).T
    indices = np.round(np.linalg.inv(M.T @ M) @ M.T @ P_rel).astype(int)

    n1, n2 = indices[0, :], indices[1, :]

    # Check fit quality
    if np.max(np.linalg.norm(points - ((M @ indices).T + origin), axis=1)) > nn_dist * 0.1:
        return None, None, None, None, None

    n1_min, n2_min = np.min(n1), np.min(n2)
    dim1, dim2 = np.max(n1) - n1_min + 1, np.max(n2) - n2_min + 1

    x_grid = np.full((dim1, dim2), np.nan)
    y_grid = np.full((dim1, dim2), np.nan)
    z_grid = np.full((dim1, dim2), np.nan)

    x_grid[n1 - n1_min, n2 - n2_min] = points[:, 0]
    y_grid[n1 - n1_min, n2 - n2_min] = points[:, 1]
    z_grid[n1 - n1_min, n2 - n2_min] = points[:, 2]

    return x_grid, y_grid, z_grid, a1, a2




def reconstruct_active_slip_planes(active_s, node_on_slip_sys, XX, YY, ZZ, box_size, min_num):
    """
    Reconstructs slip planes.
    Includes STRICT filtering to prevent massive NaN-filled arrays.
    """
    active_map = {}
    if isinstance(active_s, np.ndarray):
        rows, cols = active_s.shape
        for p in range(rows):
            for d in range(cols):
                if active_s[p, d] == 1: active_map.setdefault(p, set()).add(d)
    else:
        for sys in active_s:
            active_map.setdefault(int(sys[0]), set()).add(int(sys[1]))

    reconstructed_nodes = []
    Lx, Ly, Lz = box_size

    for (p_type, p_stack_id), indices_list in node_on_slip_sys.items():
        # --- KEY CHECK: Is this plane type active? ---
        if p_type not in active_map: continue

        ids = np.array(indices_list)

        # ---------------------------------------------------------
        # FIX IS HERE: Filter based on RAW ATOM COUNT first.
        # If the plane is defined by fewer than 100 atoms, skip it immediately.
        # ---------------------------------------------------------


        u, v, w = ids[:, 0], ids[:, 1], ids[:, 2]
        points = np.column_stack((XX[u, v, w], YY[u, v, w], ZZ[u, v, w]))

        x_grid, y_grid, z_grid, a1, a2 = structured_mesh_from_fcc_plane(points)
        if x_grid is None: continue

        # --- STRICT BOX FILTERING ---
        # Set anything outside the box to NaN
        out_mask = (x_grid < 0) | (x_grid > Lx) | (y_grid < 0) | (y_grid > Ly) | (z_grid < 0) | (z_grid > Lz)
        x_grid[out_mask] = np.nan
        y_grid[out_mask] = np.nan
        z_grid[out_mask] = np.nan

        valid_mask = ~np.isnan(x_grid)
        valid_count = np.sum(valid_mask)

        # 1. Minimum Mesh Node Count Check
        # We keep this as a secondary check. Even if we had >100 atoms,
        # if they are all outside the box (valid_count=0), we should skip.
        if valid_count < min_num: continue

        # 2. Crop Empty Rows/Cols (Bounding Box Optimization)
        rows_with_data = np.any(valid_mask, axis=1)
        cols_with_data = np.any(valid_mask, axis=0)

        ix_rows = np.where(rows_with_data)[0]
        ix_cols = np.where(cols_with_data)[0]

        x_grid = x_grid[np.ix_(ix_rows, ix_cols)]
        y_grid = y_grid[np.ix_(ix_rows, ix_cols)]
        z_grid = z_grid[np.ix_(ix_rows, ix_cols)]

        reconstructed_nodes.append({
            "x_grid": x_grid, "y_grid": y_grid, "z_grid": z_grid,
            "a_1": a1, "a_2": a2,
            "slip_plane_normal_id": p_type,
            "slip_plane_stack_id": p_stack_id,
            "active_slip_system_ids": list(active_map[p_type])
        })

    return reconstructed_nodes


# =============================================================================
# 6. DISLOCATION LOOP GENERATION
# =============================================================================

def generate_dislocation_loops_per_system(active_s, node_on_slip_sys, nodes_active, XX, YY, ZZ, b_vecs,
                                          target_density, min_size_m, max_size_m, b, box_length_m):
    generated_loops = []
    valid_mesh_points = {}

    # Map valid points from ALREADY FILTERED nodes_active
    for node in nodes_active:
        n_id, s_id = int(node['slip_plane_normal_id']), int(node['slip_plane_stack_id'])
        X, Y, Z = node['x_grid'], node['y_grid'], node['z_grid']
        valid_indices = np.argwhere(~np.isnan(X))
        if len(valid_indices) > 0:
            coords = np.column_stack((X[valid_indices[:, 0], valid_indices[:, 1]],
                                      Y[valid_indices[:, 0], valid_indices[:, 1]],
                                      Z[valid_indices[:, 0], valid_indices[:, 1]]))
            valid_mesh_points[(n_id, s_id)] = coords

    target_len = target_density * (box_length_m ** 3)
    print(f"--- Generating Loops (Target Length: {target_len:.2e} m) ---")

    for n_idx in range(4):
        for d_idx in range(3):
            # Only proceed if system is active
            if active_s[n_idx, d_idx] == 1:
                # Find stacks that belong to this normal AND are in our valid map
                available_stacks = [k[1] for k in valid_mesh_points.keys() if k[0] == n_idx]
                if not available_stacks: continue

                curr_len = 0.0
                n_vec = b_vecs[n_idx, d_idx, 0, :] / np.linalg.norm(b_vecs[n_idx, d_idx, 0, :])
                b_vec = b_vecs[n_idx, d_idx, 1, :]

                # Local basis for loop drawing
                helper = np.array([0, 0, 1]) if np.abs(n_vec[2]) < 0.9 else np.array([1, 0, 0])
                u_vec = np.cross(n_vec, helper);
                u_vec /= np.linalg.norm(u_vec)
                v_vec = np.cross(n_vec, u_vec);
                v_vec /= np.linalg.norm(v_vec)

                while curr_len < target_len:
                    stack_id = random.choice(available_stacks)
                    valid_coords = valid_mesh_points[(n_idx, stack_id)]
                    center = valid_coords[random.randint(0, len(valid_coords) - 1)]

                    a_m, c_m = random.uniform(min_size_m, max_size_m), random.uniform(min_size_m, max_size_m)
                    rot = random.uniform(0, 2 * np.pi)
                    theta = np.linspace(0, 2 * np.pi, 61)

                    x_loc = a_m * np.cos(theta)
                    y_loc = c_m * np.sin(theta)
                    x_rot = x_loc * np.cos(rot) - y_loc * np.sin(rot)
                    y_rot = x_loc * np.sin(rot) + y_loc * np.cos(rot)

                    loop_seg = center + np.outer(x_rot, u_vec) + np.outer(y_rot, v_vec)
                    curr_len += np.sum(np.linalg.norm(loop_seg[1:] - loop_seg[:-1], axis=1))

                    generated_loops.append({
                        "normal_id": int(n_idx), "slip_system_id": int(d_idx),
                        "stack_id": int(stack_id), "segments": loop_seg,
                        "burgers_vector": b_vec
                    })
                print(f"  > System ({n_idx}, {d_idx}) Generated {curr_len:.2e} m")

    return generated_loops


# =============================================================================
# 7. DENSITY MAPPING & SMEARING
# =============================================================================

def compute_density_and_theta_maps(generated_loops, nodes_active, b, nthetaintervals, std_dev_mult=600, cutoff_mult=5):
    print(f"\n--- Computing Density Maps ({nthetaintervals} intervals) ---")
    sigma = std_dev_mult * b
    cutoff = cutoff_mult * sigma
    norm_factor = 1.0 / (2 * np.pi * sigma ** 2)
    two_sigma_sq = 2 * sigma ** 2

    mesh_structs = {}
    for mesh in nodes_active:
        n, s = int(mesh['slip_plane_normal_id']), int(mesh['slip_plane_stack_id'])
        X, Y, Z = mesh['x_grid'], mesh['y_grid'], mesh['z_grid']
        a1, a2 = mesh['a_1'], mesh['a_2']
        p_norm = np.cross(a1, a2);
        p_norm /= np.linalg.norm(p_norm)

        valid = ~np.isnan(X)
        coords = np.column_stack((X[valid], Y[valid], Z[valid]))
        if len(coords) == 0: continue

        mesh_structs[(n, s)] = {
            'tree': cKDTree(coords), 'indices': np.argwhere(valid),
            'shape': X.shape, 'normal': p_norm
        }

    QQ = {}
    for idx, loop in enumerate(generated_loops):
        key = (int(loop['normal_id']), int(loop['stack_id']))
        if key not in mesh_structs: continue

        struct = mesh_structs[key]
        tree, indices, shape, n_vec = struct['tree'], struct['indices'], struct['shape'], struct['normal']

        d_map = np.zeros((shape[0], shape[1], nthetaintervals))
        pts = loop['segments']
        b_vec = loop['burgers_vector']
        b_unit = b_vec / np.linalg.norm(b_vec)

        vecs = pts[1:] - pts[:-1]
        lens = np.linalg.norm(vecs, axis=1)
        centers = (pts[1:] + pts[:-1]) / 2.0

        for i, center in enumerate(centers):
            if lens[i] == 0: continue
            t_unit = vecs[i] / lens[i]

            # Angle calc
            cos_t = np.dot(b_unit, t_unit)
            sin_t = np.dot(np.cross(b_unit, t_unit), n_vec)
            deg = np.degrees(np.arctan2(sin_t, cos_t))
            if deg < 0: deg += 360
            bin_idx = min(int(deg / (360 / nthetaintervals)), nthetaintervals - 1)

            # Spatial Smearing
            neighbors = tree.query_ball_point(center, cutoff)
            if not neighbors: continue

            w = (lens[i] * norm_factor) * np.exp(-np.sum((tree.data[neighbors] - center) ** 2, axis=1) / two_sigma_sq)

            for k, t_idx in enumerate(neighbors):
                r, c = indices[t_idx]
                d_map[r, c, bin_idx] += w[k]

        QQ[idx] = {
            "density_map": d_map, "burgers_vector": b_vec,
            "normal_id": key[0], "stack_id": key[1], "slip_system_id": int(loop['slip_system_id'])
        }
    return QQ


def smear_configurational_density(QQ_input, nthetaintervals, std_dev_rad=(5 * np.pi / 180)):
    print(f"\n--- Smearing Theta Space (std={std_dev_rad:.4f} rad) ---")
    d_theta = 2 * np.pi / nthetaintervals
    n_dist = int(np.ceil(5 * std_dev_rad / d_theta))
    k_indices = np.arange(-n_dist, n_dist + 1)
    kernel = np.exp(-((k_indices * d_theta) ** 2) / (2 * std_dev_rad ** 2))
    kernel /= np.sum(kernel)

    QQ_smeared = {}
    for key, data in QQ_input.items():
        raw = data['density_map']
        smeared = np.zeros_like(raw)

        for t in range(nthetaintervals):
            if np.sum(raw[:, :, t]) == 0: continue
            for k_idx, off in enumerate(k_indices):
                t_tgt = (t + off) % nthetaintervals
                smeared[:, :, t_tgt] += raw[:, :, t] * kernel[k_idx]

        QQ_smeared[key] = data.copy()
        QQ_smeared[key]['density_map'] = smeared
    return QQ_smeared


def normalize_density_to_target(QQ, nodes_active, target_density):
    print(f"\n--- Normalizing Density to Target: {target_density:.2e} ---")

    # 1. Count Valid Nodes per System
    total_nodes = {}
    for mesh in nodes_active:
        n_id = int(mesh['slip_plane_normal_id'])
        valid_count = np.sum(~np.isnan(mesh['x_grid']))
        for sys_id in mesh['active_slip_system_ids']:
            k = (n_id, int(sys_id))
            total_nodes[k] = total_nodes.get(k, 0) + valid_count

    # 2. Sum Density
    total_dens = {}
    for d in QQ.values():
        k = (int(d['normal_id']), int(d['slip_system_id']))
        total_dens[k] = total_dens.get(k, 0.0) + np.sum(d['density_map'])

    # 3. Scale
    QQ_norm = QQ.copy()
    for d in QQ_norm.values():
        k = (int(d['normal_id']), int(d['slip_system_id']))
        nodes = total_nodes.get(k, 0)
        curr = total_dens.get(k, 0)

        factor = (target_density / (curr / nodes)) if (nodes > 0 and curr > 0) else 1.0
        d['density_map'] *= factor

    return QQ_norm