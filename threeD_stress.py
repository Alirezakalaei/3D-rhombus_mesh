import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree


# =============================================================================
# HELPER FUNCTIONS (TENSORS)
# =============================================================================

def get_elastic_constants(mu, nu):
    """Constructs the isotropic elastic stiffness tensor C_ijkl."""
    C = np.zeros((3, 3, 3, 3))
    lam = 2 * mu * nu / (1 - 2 * nu)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    delta_ij = 1.0 if i == j else 0.0
                    delta_kl = 1.0 if k == l else 0.0
                    delta_ik = 1.0 if i == k else 0.0
                    delta_jl = 1.0 if j == l else 0.0
                    delta_il = 1.0 if i == l else 0.0
                    delta_jk = 1.0 if j == k else 0.0

                    C[i, j, k, l] = lam * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
    return C


def get_epsilon_delta():
    """Returns the Levi-Civita tensor and Kronecker delta."""
    epsilon = np.zeros((3, 3, 3))
    delta = np.eye(3)

    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0

    return epsilon, delta


# =============================================================================
# GND AND TANGENT COMPUTATION
# =============================================================================

@njit(fastmath=True)
def compute_gnd_and_xi_single_plane(density_map, theta_array):
    """
    Compute GND density and tangent vector xi for a SINGLE plane/stack.
    """
    nx, ny, n_thet = density_map.shape
    GND = np.zeros((nx, ny))
    xi_local = np.zeros((nx, ny, 3))

    cos_theta = np.cos(theta_array)
    sin_theta = np.sin(theta_array)

    for i in range(nx):
        for j in range(ny):
            vec_x = 0.0
            vec_y = 0.0

            for k in range(n_thet):
                rho = density_map[i, j, k]
                vec_x += rho * cos_theta[k]
                vec_y += rho * sin_theta[k]

            mag = np.sqrt(vec_x ** 2 + vec_y ** 2)
            GND[i, j] = mag

            if mag > 1e-12:
                xi_local[i, j, 0] = vec_x / mag
                xi_local[i, j, 1] = vec_y / mag
                xi_local[i, j, 2] = 0.0
            else:
                xi_local[i, j, 0] = 0.0
                xi_local[i, j, 1] = 0.0
                xi_local[i, j, 2] = 0.0

    return GND, xi_local


# =============================================================================
# PARALLEL KERNEL
# =============================================================================

@njit(parallel=True, fastmath=True)
def compute_stress_batch(
        target_coords,  # (N_targets, 3)
        neighbor_offsets,  # (N_targets + 1,)
        neighbor_indices,  # (Total_interactions,)
        source_coords,  # (N_sources, 3)
        source_densities,  # (N_sources,)
        source_normals,  # (N_sources, 3)
        source_tangents,  # (N_sources, 3) - This is xi (vector)
        b_prime,  # (3,)
        C, epsilon_tens, delta,
        mu, nu, a
):
    """
    Computes the scalar resolved shear stress at target points.
    Performs the projection (dot product) of the Mura vector onto the source tangent vector.
    """
    n_targets = target_coords.shape[0]
    results = np.zeros(n_targets)

    prefactor_base = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))

    # Pre-calculate constants to avoid re-computing in inner loop
    term1_const = 6.0 * a ** 2 * (1.0 - nu)
    term2_const = (3.0 - 4.0 * nu)

    # Parallel loop over targets
    for t_idx in prange(n_targets):

        t_pos = target_coords[t_idx]

        # Get neighbors for this target
        start = neighbor_offsets[t_idx]
        end = neighbor_offsets[t_idx + 1]

        local_val = 0.0  # Scalar accumulator

        for k in range(start, end):
            src_idx = neighbor_indices[k]

            s_pos = source_coords[src_idx]

            # Distance check
            dx = t_pos[0] - s_pos[0]
            dy = t_pos[1] - s_pos[1]
            dz = t_pos[2] - s_pos[2]
            dist_sq = dx * dx + dy * dy + dz * dz

            # Safety check for regularization
            R2 = dist_sq + a ** 2
            if R2 < 1e-20: continue  # Prevent division by zero

            rho = source_densities[src_idx]
            n_alpha = source_normals[src_idx]
            xi_alpha = source_tangents[src_idx]

            # --- Inline Mura G Tensor Calculation ---
            R = np.sqrt(R2)
            invR3 = 1.0 / (R2 * R)
            invR5 = 1.0 / (R2 * R2 * R)

            g_h_0 = 0.0
            g_h_1 = 0.0
            g_h_2 = 0.0

            # Unrolled loops for specific tensor contractions would be faster, 
            # but we keep the structure generic for correctness with the C tensor.

            # We iterate h (0,1,2)
            for h in range(3):
                total_h = 0.0

                # Iterate p, q
                for p in range(3):
                    np_p = n_alpha[p]
                    if np_p == 0: continue

                    for q in range(3):
                        vec_q = xi_alpha[q]
                        if vec_q == 0: continue

                        # Iterate r, s
                        for r_i in range(3):
                            for s in range(3):
                                C_pqrs = C[p, q, r_i, s]
                                if abs(C_pqrs) < 1e-12: continue

                                coeff = np_p * vec_q * C_pqrs * prefactor_base

                                # Iterate j
                                for j in range(3):
                                    eps_jsh = epsilon_tens[j, s, h]
                                    if eps_jsh == 0: continue

                                    # Iterate i
                                    for i in range(3):
                                        bi_p = b_prime[i]
                                        if bi_p == 0: continue

                                        # Iterate k, l
                                        for k_idx in range(3):
                                            x_k = dx if k_idx == 0 else (dy if k_idx == 1 else dz)

                                            for l in range(3):
                                                x_l = dx if l == 0 else (dy if l == 1 else dz)
                                                x_r = dx if r_i == 0 else (dy if r_i == 1 else dz)

                                                Cijkl = C[i, j, k_idx, l]
                                                if abs(Cijkl) < 1e-12: continue

                                                delta_rk = delta[r_i, k_idx]
                                                delta_rl = delta[r_i, l]
                                                delta_kl = delta[k_idx, l]

                                                term1 = delta_rk * term1_const * x_l * invR5
                                                term2 = term2_const * delta_rk * x_l * invR3
                                                term3 = 3.0 * x_r * x_k * x_l * invR5
                                                term4 = - (delta_rl * x_k + delta_kl * x_r) * invR3

                                                bracket = term1 + term2 + term3 + term4
                                                total_h += coeff * eps_jsh * bi_p * Cijkl * bracket

                if h == 0:
                    g_h_0 = total_h
                elif h == 1:
                    g_h_1 = total_h
                else:
                    g_h_2 = total_h

            # Dot Product with xi_alpha
            dot_val = g_h_0 * xi_alpha[0] + g_h_1 * xi_alpha[1] + g_h_2 * xi_alpha[2]
            local_val += rho * dot_val

        results[t_idx] = -local_val

    return results


# =============================================================================
# MAIN COMPUTATION FUNCTION
# =============================================================================

def compute_interaction_stress_mura(QQ_smeared, C_tensor, eps_tensor, delta_tensor, nodes_active, b_vecs, b, mu, nu,
                                    a_param,
                                    cut_off_dist=3000, min_cut_off_density=1e9):
    """
    Computes Mura interaction stress using computed GND and Tangent vectors (xi).
    Parallelized and optimized. Returns a scalar stress field per slip system.
    """
    print(f"\n--- Computing Mura Interaction Stress (a={a_param:.2e}, cutoff={cut_off_dist:.2e}) ---")

    # 1. PREPARE SOURCES
    # ------------------
    source_coords_list = []
    source_dens_list = []
    source_norm_list = []
    source_tan_list = []

    # Map (n_id, s_id) -> (X, Y, Z) for fast lookup
    mesh_lookup = {}
    for mesh in nodes_active:
        nid = int(mesh['slip_plane_normal_id'])
        sid = int(mesh['slip_plane_stack_id'])
        mesh_lookup[(nid, sid)] = (mesh['x_grid'], mesh['y_grid'], mesh['z_grid'])

    # Helper: Get Normal Vectors from b_vecs
    plane_normals = []
    for i in range(4):
        n = b_vecs[i, 0, 0, :]
        plane_normals.append(n / np.linalg.norm(n))

    # Pre-calculate Theta Array
    if not QQ_smeared:
        print("Error: QQ_smeared is empty.")
        return {}

    first_key = list(QQ_smeared.keys())[0]
    n_theta = QQ_smeared[first_key]['density_map'].shape[2]
    theta_arr = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    print("  > Extracting Sources (GND & Tangents)...")

    for key, data in QQ_smeared.items():
        n_id = data['normal_id']
        s_id = data['stack_id']
        n_vec_global = plane_normals[n_id]

        if (n_id, s_id) not in mesh_lookup: continue
        X, Y, Z = mesh_lookup[(n_id, s_id)]

        # --- ROBUST BASIS VECTOR CALCULATION ---
        # Find valid grid points to define u_vec and v_vec.
        # We cannot assume X[0,0] is valid.
        u_vec = None
        v_vec = None

        # Search for a valid horizontal pair for u_vec
        # Look for (r, c) and (r, c+1) that are both valid
        rows_valid, cols_valid = np.where(~np.isnan(X))

        # Try to find u_vec (along columns)
        for idx in range(len(rows_valid)):
            r, c = rows_valid[idx], cols_valid[idx]
            if c + 1 < X.shape[1] and not np.isnan(X[r, c + 1]):
                u_vec = np.array([X[r, c + 1] - X[r, c], Y[r, c + 1] - Y[r, c], Z[r, c + 1] - Z[r, c]])
                norm_u = np.linalg.norm(u_vec)
                if norm_u > 1e-12:
                    u_vec /= norm_u
                    break

        # Try to find v_vec (along rows)
        for idx in range(len(rows_valid)):
            r, c = rows_valid[idx], cols_valid[idx]
            if r + 1 < X.shape[0] and not np.isnan(X[r + 1, c]):
                v_vec = np.array([X[r + 1, c] - X[r, c], Y[r + 1, c] - Y[r, c], Z[r + 1, c] - Z[r, c]])
                norm_v = np.linalg.norm(v_vec)
                if norm_v > 1e-12:
                    v_vec /= norm_v
                    break

        # Fallback if grid is too sparse or 1D
        if u_vec is None or v_vec is None:
            # Construct arbitrary orthogonal basis from normal
            if np.abs(n_vec_global[2]) < 0.9:
                helper = np.array([0, 0, 1])
            else:
                helper = np.array([1, 0, 0])
            u_vec = np.cross(n_vec_global, helper)
            u_vec /= np.linalg.norm(u_vec)
            v_vec = np.cross(n_vec_global, u_vec)
            v_vec /= np.linalg.norm(v_vec)

        # ---------------------------------------

        # Compute GND and Local Tangent
        GND_map, xi_local_map = compute_gnd_and_xi_single_plane(data['density_map'], theta_arr)

        # Filter by cutoff density AND valid coordinates
        # Crucial: Ensure we don't pick up NaN coordinates even if density is somehow high
        valid_coords_mask = ~np.isnan(X)
        high_density_mask = GND_map > min_cut_off_density
        combined_mask = valid_coords_mask & high_density_mask

        rows, cols = np.where(combined_mask)

        if len(rows) == 0: continue

        # Vectorized extraction for this mesh
        densities = GND_map[rows, cols]

        # Positions
        pos_x = X[rows, cols]
        pos_y = Y[rows, cols]
        pos_z = Z[rows, cols]
        positions = np.column_stack((pos_x, pos_y, pos_z))

        # Tangents (Rotation)
        loc_xi = xi_local_map[rows, cols]  # (N, 3)
        # global = x * u + y * v
        glob_xi = np.outer(loc_xi[:, 0], u_vec) + np.outer(loc_xi[:, 1], v_vec)

        # Normalize Tangents
        norms = np.linalg.norm(glob_xi, axis=1)
        valid_norm = norms > 1e-12

        # Initialize with zeros
        glob_xi_normalized = np.zeros_like(glob_xi)
        glob_xi_normalized[valid_norm] = glob_xi[valid_norm] / norms[valid_norm, None]

        # Append to lists
        source_coords_list.append(positions)
        source_dens_list.append(densities)
        # Normal is constant for the whole mesh
        normals = np.tile(n_vec_global, (len(densities), 1))
        source_norm_list.append(normals)
        source_tan_list.append(glob_xi_normalized)

    if not source_coords_list:
        print("  ! No sources found above density cutoff.")
        return {}

    # Flatten source arrays for Numba
    src_coords_arr = np.vstack(source_coords_list)
    src_dens_arr = np.concatenate(source_dens_list)
    src_norm_arr = np.vstack(source_norm_list)
    src_tan_arr = np.vstack(source_tan_list)

    print(f"  > Built Source Tree: {len(src_coords_arr)} sources.")
    source_tree = cKDTree(src_coords_arr)

    # 2. COMPUTE TARGETS
    # ------------------
    stress_results = {}
    print(f"  > Computing targets for {len(nodes_active)} active stacks...")

    for mesh in nodes_active:
        n_id_target = int(mesh['slip_plane_normal_id'])
        s_id_target = int(mesh['slip_plane_stack_id'])

        X_target = mesh['x_grid']
        Y_target = mesh['y_grid']
        Z_target = mesh['z_grid']

        # Identify valid grid points (Strict NaN check)
        valid_mask = ~np.isnan(X_target)
        target_indices = np.argwhere(valid_mask)  # (N_valid, 2)

        if len(target_indices) == 0: continue

        # Extract target coordinates
        t_rows = target_indices[:, 0]
        t_cols = target_indices[:, 1]
        t_pos_x = X_target[t_rows, t_cols]
        t_pos_y = Y_target[t_rows, t_cols]
        t_pos_z = Z_target[t_rows, t_cols]
        target_coords_arr = np.column_stack((t_pos_x, t_pos_y, t_pos_z))

        # QUERY TREE (Vectorized)
        neighbor_lists = source_tree.query_ball_point(target_coords_arr, cut_off_dist)

        # Flatten neighbors for Numba parallelization
        lengths = np.array([len(l) for l in neighbor_lists], dtype=np.int64)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        if offsets[-1] == 0:
            flat_indices = np.zeros(0, dtype=np.int64)
        else:
            flat_indices = np.array([item for sublist in neighbor_lists for item in sublist], dtype=np.int64)

        # Loop over slip systems
        for sys_id in range(3):
            b_prime = b_vecs[n_id_target, sys_id, 1, :]

            # CALL PARALLEL KERNEL
            if len(flat_indices) > 0:
                # Returns 1D array of scalars
                computed_stress_flat = compute_stress_batch(
                    target_coords_arr,
                    offsets,
                    flat_indices,
                    src_coords_arr,
                    src_dens_arr,
                    src_norm_arr,
                    src_tan_arr,
                    b_prime,
                    C_tensor, eps_tensor, delta_tensor,
                    mu, nu, a_param
                )
            else:
                computed_stress_flat = np.zeros(len(target_coords_arr))

            # Remap flat results back to (nx, ny) scalar grid
            nx, ny = X_target.shape

            # Stress field is now 2D (Scalar field)
            stress_field = np.zeros((nx, ny))

            # Fill valid points, leave NaNs as 0.0 (or could set to NaN if preferred)
            stress_field[t_rows, t_cols] = computed_stress_flat

            # Store results
            result_key = (n_id_target, s_id_target, sys_id)
            stress_results[result_key] = {
                "stress_field": stress_field,
                "slip_plane_normal_id": n_id_target,
                "slip_plane_stack_id": s_id_target,
                "slip_system_id": sys_id,
                "burgers_vector": b_prime
            }

    print("  > Computation Complete.")
    return stress_results