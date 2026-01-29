import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree


# =============================================================================
# OPTIMIZED NUMBA KERNEL
# =============================================================================
@njit(parallel=True, fastmath=True)
def compute_stress_cai_flattened(target_coords, target_burgers, target_normals,
                                 neighbor_offsets, neighbor_indices,
                                 source_coords, source_densities, source_C_vecs,
                                 mu, nu, a, dv):
    """
    Optimized Stress Computation.
    Changes:
    - Removed source_burgers and source_tangents from arguments.
    - Added source_C_vecs (The cross product b x xi, pre-calculated).
    - Removed cross-product calculation from the inner loop.
    """
    n_targets = target_coords.shape[0]
    results = np.zeros(n_targets, dtype=np.float64)

    # Pre-calculate constants
    factor1 = mu / (8.0 * np.pi)
    factor2 = mu / (4.0 * np.pi * (1.0 - nu))
    a2 = a * a

    for t_idx in prange(n_targets):
        t_pos = target_coords[t_idx]
        t_b = target_burgers[t_idx]
        t_n = target_normals[t_idx]

        start = neighbor_offsets[t_idx]
        end = neighbor_offsets[t_idx + 1]

        # Local stress tensor accumulators
        sigma_xx = 0.0
        sigma_yy = 0.0
        sigma_zz = 0.0
        sigma_xy = 0.0
        sigma_yz = 0.0
        sigma_zx = 0.0

        for k in range(start, end):
            src_idx = neighbor_indices[k]
            rho = source_densities[src_idx]

            # Skip insignificant densities to save compute
            if rho <= 1e-12:
                continue

            s_pos = source_coords[src_idx]

            # Vector R = r - r'
            dx = t_pos[0] - s_pos[0]
            dy = t_pos[1] - s_pos[1]
            dz = t_pos[2] - s_pos[2]

            R2 = dx * dx + dy * dy + dz * dz
            Ra2 = R2 + a2
            Ra = np.sqrt(Ra2)
            Ra3 = Ra2 * Ra
            Ra5 = Ra3 * Ra2

            # Retrieve pre-calculated Cross Product (b x xi)
            # This saves 6 muls and 3 subs per interaction
            C_vec = source_C_vecs[src_idx]
            C_x = C_vec[0]
            C_y = C_vec[1]
            C_z = C_vec[2]

            # Derivatives of Green's function parts
            lap_Ra = (2.0 * R2 + 3.0 * a2) / Ra3
            grad_lap_Ra_factor = (-2.0 * R2 - 5.0 * a2) / Ra5

            d_i_lap_Ra_x = dx * grad_lap_Ra_factor
            d_i_lap_Ra_y = dy * grad_lap_Ra_factor
            d_i_lap_Ra_z = dz * grad_lap_Ra_factor

            # Term 1 Components
            T1_xx = 2.0 * C_x * d_i_lap_Ra_x
            T1_yy = 2.0 * C_y * d_i_lap_Ra_y
            T1_zz = 2.0 * C_z * d_i_lap_Ra_z
            T1_xy = C_x * d_i_lap_Ra_y + C_y * d_i_lap_Ra_x
            T1_yz = C_y * d_i_lap_Ra_z + C_z * d_i_lap_Ra_y
            T1_zx = C_z * d_i_lap_Ra_x + C_x * d_i_lap_Ra_z

            # Term 2 Components
            C_dot_x = C_x * dx + C_y * dy + C_z * dz
            invRa3 = 1.0 / Ra3
            invRa5 = 1.0 / Ra5

            # Common factor for D3 terms
            Three_C_dot_x_invRa5 = 3.0 * C_dot_x * invRa5

            D3_xx = - (C_dot_x + 2.0 * C_x * dx) * invRa3 + Three_C_dot_x_invRa5 * dx * dx
            D3_yy = - (C_dot_x + 2.0 * C_y * dy) * invRa3 + Three_C_dot_x_invRa5 * dy * dy
            D3_zz = - (C_dot_x + 2.0 * C_z * dz) * invRa3 + Three_C_dot_x_invRa5 * dz * dz
            D3_xy = - (C_x * dy + C_y * dx) * invRa3 + Three_C_dot_x_invRa5 * dx * dy
            D3_yz = - (C_y * dz + C_z * dy) * invRa3 + Three_C_dot_x_invRa5 * dy * dz
            D3_zx = - (C_z * dx + C_x * dz) * invRa3 + Three_C_dot_x_invRa5 * dz * dx

            C_dot_grad_lap = C_x * d_i_lap_Ra_x + C_y * d_i_lap_Ra_y + C_z * d_i_lap_Ra_z

            T2_xx = D3_xx - C_dot_grad_lap
            T2_yy = D3_yy - C_dot_grad_lap
            T2_zz = D3_zz - C_dot_grad_lap
            T2_xy = D3_xy
            T2_yz = D3_yz
            T2_zx = D3_zx

            scale = rho * dv
            sigma_xx += scale * (factor1 * T1_xx + factor2 * T2_xx)
            sigma_yy += scale * (factor1 * T1_yy + factor2 * T2_yy)
            sigma_zz += scale * (factor1 * T1_zz + factor2 * T2_zz)
            sigma_xy += scale * (factor1 * T1_xy + factor2 * T2_xy)
            sigma_yz += scale * (factor1 * T1_yz + factor2 * T2_yz)
            sigma_zx += scale * (factor1 * T1_zx + factor2 * T2_zx)

        # Project stress tensor onto target slip system (RSS)
        # RSS = (sigma . n) . b
        t_x = sigma_xx * t_n[0] + sigma_xy * t_n[1] + sigma_zx * t_n[2]
        t_y = sigma_xy * t_n[0] + sigma_yy * t_n[1] + sigma_yz * t_n[2]
        t_z = sigma_zx * t_n[0] + sigma_yz * t_n[1] + sigma_zz * t_n[2]

        results[t_idx] = t_x * t_b[0] + t_y * t_b[1] + t_z * t_b[2]

    return results


@njit(fastmath=True)
def compute_gnd_and_xi_single_plane(density_map, theta_array):
    nx, ny, n_thet = density_map.shape
    GND = np.zeros((nx, ny))
    xi_local = np.zeros((nx, ny, 3))
    cos_theta = np.cos(theta_array)
    sin_theta = np.sin(theta_array)

    for i in range(nx):
        for j in range(ny):
            # Check if density exists at this spatial point (sum check or first bin check)
            # Assuming if first bin is nan, all are nan
            if np.isnan(density_map[i, j, 0]):
                GND[i, j] = 0.0
                continue

            vec_x = 0.0
            vec_y = 0.0
            total_rho = 0.0

            for k in range(n_thet):
                rho = density_map[i, j, k]
                vec_x += rho * cos_theta[k]
                vec_y += rho * sin_theta[k]
                total_rho += rho  # Optional: if GND is sum of magnitudes

            mag = np.sqrt(vec_x ** 2 + vec_y ** 2)
            GND[i, j] = mag

            if mag > 1e-12:
                xi_local[i, j, 0] = vec_x / mag
                xi_local[i, j, 1] = vec_y / mag

    return GND, xi_local


def get_elastic_constants(mu, nu):
    return np.zeros((3, 3, 3, 3))


def get_epsilon_delta():
    return np.zeros((3, 3, 3)), np.eye(3)


# =============================================================================
# MAIN COMPUTATION WRAPPER
# =============================================================================
def compute_interaction_stress_mura(QQ_smeared, C_tensor, eps_tensor, delta_tensor,
                                    nodes_active, b_vecs, b, mu, nu,
                                    a_param, dvol, cut_off_dist=3000, min_cut_off_density=1e9):
    print(f"\n--- Computing Interaction Stress (Strict Active-Only) ---")

    # =========================================================
    # PHASE 1: FLATTEN SOURCES
    # =========================================================
    print("  > Phase 1: Flattening Sources...")
    source_coords_list = []
    source_dens_list = []
    source_C_list = []  # Stores b x xi

    plane_normals = []
    for i in range(4):
        n = b_vecs[i, 0, 0, :]
        plane_normals.append(n / np.linalg.norm(n))

    mesh_lookup = {}
    for mesh in nodes_active:
        nid = int(mesh['slip_plane_normal_id'])
        sid = int(mesh['slip_plane_stack_id'])
        mesh_lookup[(nid, sid)] = (mesh['x_grid'], mesh['y_grid'], mesh['z_grid'])

    if not QQ_smeared:
        return {}

    first_key = list(QQ_smeared.keys())[0]
    n_theta = QQ_smeared[first_key]['density_map'].shape[2]
    theta_arr = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    for key, data in QQ_smeared.items():
        n_id = data['normal_id']
        s_id = data['stack_id']
        n_vec_global = plane_normals[n_id]
        b_vec_global = data['burgers_vector']

        if (n_id, s_id) not in mesh_lookup:
            continue

        X, Y, Z = mesh_lookup[(n_id, s_id)]

        # Determine local basis vectors for the plane
        helper = np.array([0, 0, 1]) if np.abs(n_vec_global[2]) < 0.9 else np.array([1, 0, 0])
        u_vec = np.cross(n_vec_global, helper)
        u_vec /= np.linalg.norm(u_vec)
        v_vec = np.cross(n_vec_global, u_vec)
        v_vec /= np.linalg.norm(v_vec)

        # Compute GND magnitude and local tangent direction
        GND_map, xi_local_map = compute_gnd_and_xi_single_plane(data['density_map'], theta_arr)

        valid_coords_mask = (~np.isnan(X)) & (~np.isnan(Y)) & (~np.isnan(Z))
        valid_density_mask = (GND_map > min_cut_off_density) & (GND_map > 0.0)
        combined_mask = valid_coords_mask & valid_density_mask

        rows, cols = np.where(combined_mask)
        if len(rows) == 0:
            continue

        densities = GND_map[rows, cols]
        positions = np.column_stack((X[rows, cols], Y[rows, cols], Z[rows, cols]))

        # Convert local 2D tangent to global 3D tangent
        loc_xi = xi_local_map[rows, cols]
        glob_xi = np.outer(loc_xi[:, 0], u_vec) + np.outer(loc_xi[:, 1], v_vec)

        # Normalize tangents
        norms = np.linalg.norm(glob_xi, axis=1)
        valid_norm = norms > 1e-12
        glob_xi[valid_norm] = glob_xi[valid_norm] / norms[valid_norm, None]
        glob_xi[~valid_norm] = 0.0

        # --- OPTIMIZATION: PRE-COMPUTE CROSS PRODUCT (b x xi) ---
        # b_vec_global is constant for this whole loop iteration.
        # glob_xi varies per point.
        # We compute the cross product here to avoid doing it inside the Numba loop.
        # Broadcast b_vec to match shape of glob_xi
        b_broadcast = np.tile(b_vec_global, (len(densities), 1))
        C_vectors = np.cross(b_broadcast, glob_xi)

        source_coords_list.append(positions)
        source_dens_list.append(densities)
        source_C_list.append(C_vectors)

    if not source_coords_list:
        print("  ! No valid sources found.")
        return {}

    src_coords_arr = np.vstack(source_coords_list).astype(np.float64)
    src_dens_arr = np.concatenate(source_dens_list).astype(np.float64)
    src_C_arr = np.vstack(source_C_list).astype(np.float64)

    print(f"    > Total Source Points: {len(src_coords_arr)}")
    source_tree = cKDTree(src_coords_arr)

    # =========================================================
    # PHASE 2: FLATTEN TARGETS (STRICTLY ACTIVE ONLY)
    # =========================================================
    print("  > Phase 2: Flattening Targets...")
    target_coords_list = []
    target_burgers_list = []
    target_normals_list = []
    reconstruction_map = []
    current_idx = 0

    for mesh in nodes_active:
        n_id = int(mesh['slip_plane_normal_id'])
        s_id = int(mesh['slip_plane_stack_id'])
        X = mesh['x_grid']
        Y = mesh['y_grid']
        Z = mesh['z_grid']

        valid_mask = (~np.isnan(X)) & (~np.isnan(Y)) & (~np.isnan(Z))
        rows, cols = np.where(valid_mask)
        if len(rows) == 0:
            continue

        mesh_coords = np.column_stack((X[rows, cols], Y[rows, cols], Z[rows, cols]))
        n_points = len(mesh_coords)

        n_vec = plane_normals[n_id]
        n_vec_repeated = np.tile(n_vec, (n_points, 1))

        # Retrieve active systems for this plane
        active_systems = mesh.get('active_slip_system_ids', [])

        for sys_id in active_systems:
            sys_id = int(sys_id)
            b_vec = b_vecs[n_id, sys_id, 1, :]
            b_vec_repeated = np.tile(b_vec, (n_points, 1))

            target_coords_list.append(mesh_coords)
            target_normals_list.append(n_vec_repeated)
            target_burgers_list.append(b_vec_repeated)

            reconstruction_map.append({
                'key': (n_id, s_id, sys_id),
                'shape': X.shape, 'rows': rows, 'cols': cols,
                'start': current_idx, 'end': current_idx + n_points,
                'burgers_vector': b_vec
            })
            current_idx += n_points

    if not target_coords_list:
        return {}

    tgt_coords_arr = np.vstack(target_coords_list).astype(np.float64)
    tgt_norm_arr = np.vstack(target_normals_list).astype(np.float64)
    tgt_burg_arr = np.vstack(target_burgers_list).astype(np.float64)
    print(f"    > Total Calculation Points: {len(tgt_coords_arr)}")

    # =========================================================
    # PHASE 3: COMPUTE
    # =========================================================
    print("  > Phase 3: Spatial Query & Numba Execution...")

    # KDTree Query
    neighbor_lists = source_tree.query_ball_point(tgt_coords_arr, cut_off_dist)

    # Flatten neighbor lists for Numba
    lengths = np.array([len(l) for l in neighbor_lists], dtype=np.int64)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)

    if offsets[-1] > 0:
        flat_indices = np.array([item for sublist in neighbor_lists for item in sublist], dtype=np.int64)
    else:
        flat_indices = np.zeros(0, dtype=np.int64)

    # Call Optimized Numba Function
    # Note: Passing src_C_arr instead of burgers/tangents
    flat_rss_results = compute_stress_cai_flattened(
        tgt_coords_arr, tgt_burg_arr, tgt_norm_arr,
        offsets, flat_indices,
        src_coords_arr, src_dens_arr, src_C_arr,
        mu, nu, a_param, dvol
    )

    # =========================================================
    # PHASE 4: RECONSTRUCT
    # =========================================================
    print("  > Phase 4: Reconstructing Dictionaries...")
    stress_results = {}
    for item in reconstruction_map:
        key = item['key']
        start = item['start']
        end = item['end']

        values = flat_rss_results[start:end]

        nx, ny = item['shape']
        stress_field = np.zeros((nx, ny))
        rows = item['rows']
        cols = item['cols']
        stress_field[rows, cols] = values

        stress_results[key] = {
            "stress_field": stress_field,
            "slip_plane_normal_id": key[0],
            "slip_plane_stack_id": key[1],
            "slip_system_id": key[2],
            "burgers_vector": item['burgers_vector']
        }

    print("  > Done.")
    return stress_results