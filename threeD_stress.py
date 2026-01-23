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
# MURA KERNEL (OPTIMIZED & INLINED)
# =============================================================================

@njit(fastmath=True)
def mura_g_tensor_inline(r_vec, n_alpha, xi_alpha, b_prime, C, prefactor_base, a, epsilon_tens, delta):
    """
    Computes the interaction vector g_h.
    Refactored to take precomputed constants for speed inside loops.
    """
    # Compute R terms
    R2 = r_vec[0] ** 2 + r_vec[1] ** 2 + r_vec[2] ** 2 + a ** 2
    R = np.sqrt(R2)
    invR3 = 1.0 / (R ** 3)
    invR5 = 1.0 / (R ** 5)

    g_h = np.zeros(3)

    # Tensor contraction
    for h in range(3):
        total = 0.0
        for p in range(3):
            np_p = n_alpha[p]
            for q in range(3):
                vec_q = xi_alpha[q]

                for r_idx in range(3):
                    for s in range(3):
                        # Pre-check C to avoid deep loop if zero (optimization)
                        if C[p, q, r_idx, s] == 0: continue

                        coeff = np_p * vec_q * C[p, q, r_idx, s] * prefactor_base

                        for j in range(3):
                            eps_jsh = epsilon_tens[j, s, h]
                            if eps_jsh == 0: continue

                            for i in range(3):
                                bi_prime = b_prime[i]
                                for k in range(3):
                                    x_k = r_vec[k]
                                    for l in range(3):
                                        x_l = r_vec[l]
                                        x_r = r_vec[r_idx]

                                        Cijkl = C[i, j, k, l]
                                        if Cijkl == 0: continue

                                        delta_rk = delta[r_idx, k]
                                        delta_rl = delta[r_idx, l]
                                        delta_kl = delta[k, l]

                                        term1 = 6.0 * delta_rk * a ** 2 * (
                                                    1.0 - 0.3) * x_l * invR5  # Note: nu hardcoded here? No, passed via precalc usually, fixed below.
                                        # To fix the nu issue inside this tight loop without passing nu again:
                                        # We will assume the caller handles the logic or we pass nu.
                                        # For strict correctness based on previous code:
                                        # term1 = 6.0 * delta_rk * a**2 * (1.0 - nu) * x_l * invR5
                                        # But let's stick to the math structure.
                                        # To keep it fast, we do the bracket calculation:

                                        # Re-implementing strictly:
                                        # We need (1-nu). Let's assume it's passed or derived.
                                        # Actually, let's just do the math inside the main kernel below to avoid signature mess.
                                        pass
    return g_h


# =============================================================================
# PARALLEL KERNEL
# =============================================================================

@njit(parallel=True, fastmath=True)
def compute_stress_kernel_parallel(
        target_coords,  # (N_targets, 3)
        neighbor_counts,  # (N_targets,) - How many neighbors each target has
        neighbor_indices,  # (Total_Neighbors,) - Flattened list of neighbor indices
        source_coords,  # (N_sources, 3)
        source_densities,  # (N_sources,)
        source_normals,  # (N_sources, 3)
        source_tangents,  # (N_sources, 3)
        b_prime,  # (3,)
        C,  # (3,3,3,3)
        epsilon_tens,  # (3,3,3)
        delta,  # (3,3)
        mu, nu, a
):
    """
    Parallel loop over target points to compute stress.
    """
    n_targets = target_coords.shape[0]
    stress_results = np.zeros((n_targets, 3))

    prefactor_base = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))

    # We need a pointer to where the neighbors for the current target start in the flattened list
    # Since parallel loops can't easily share a mutable counter, we usually need an offset array.
    # However, constructing the offset array is fast.

    # Construct offsets (must be done outside parallel or carefully inside)
    # Since Numba parallel doesn't support cumsum easily, we assume offsets are passed
    # OR we change strategy: The caller (Python) handles the KDTree flattening.
    # Let's assume neighbor_indices is a list of arrays, but Numba doesn't like list of arrays in parallel well.
    # BEST APPROACH FOR NUMBA: Flattened arrays with an offset array.

    # But wait, we can't easily generate offsets inside the parallel function.
    # We will modify the input to take `neighbor_offsets`.

    return stress_results  # Placeholder, see actual implementation below


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
    # CHANGE 1: Result is now a scalar array (N,), not a vector array (N, 3)
    results = np.zeros(n_targets)

    prefactor_base = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))

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
            rho = source_densities[src_idx]
            n_alpha = source_normals[src_idx]
            xi_alpha = source_tangents[src_idx]  # This is xi

            # r_vec calculation
            r_vec = np.zeros(3)
            dist_sq = 0.0
            for d in range(3):
                val = t_pos[d] - s_pos[d]
                r_vec[d] = val
                dist_sq += val * val

            if dist_sq < 1e-20: continue

            # --- Inline Mura G Tensor Calculation (Vector Output) ---
            R2 = dist_sq + a ** 2
            R = np.sqrt(R2)
            invR3 = 1.0 / (R ** 3)
            invR5 = 1.0 / (R ** 5)

            g_h = np.zeros(3)

            for h in range(3):
                total_h = 0.0
                for p in range(3):
                    np_p = n_alpha[p]
                    for q in range(3):
                        vec_q = xi_alpha[q]
                        for r_i in range(3):
                            for s in range(3):
                                # Optimization: Check C first
                                C_pqrs = C[p, q, r_i, s]
                                if abs(C_pqrs) < 1e-12: continue

                                coeff = np_p * vec_q * C_pqrs * prefactor_base

                                for j in range(3):
                                    eps_jsh = epsilon_tens[j, s, h]
                                    if eps_jsh == 0: continue

                                    for i in range(3):
                                        bi_p = b_prime[i]
                                        for k_idx in range(3):
                                            x_k = r_vec[k_idx]
                                            for l in range(3):
                                                x_l = r_vec[l]
                                                x_r = r_vec[r_i]

                                                Cijkl = C[i, j, k_idx, l]

                                                delta_rk = delta[r_i, k_idx]
                                                delta_rl = delta[r_i, l]
                                                delta_kl = delta[k_idx, l]

                                                term1 = 6.0 * delta_rk * a ** 2 * (1.0 - nu) * x_l * invR5
                                                term2 = (3.0 - 4.0 * nu) * delta_rk * x_l * invR3
                                                term3 = 3.0 * x_r * x_k * x_l * invR5
                                                term4 = - (delta_rl * x_k + delta_kl * x_r) * invR3

                                                bracket = term1 + term2 + term3 + term4
                                                total_h += coeff * eps_jsh * bi_p * Cijkl * bracket
                g_h[h] = total_h
            # --- End Inline ---

            # CHANGE 2: Perform Dot Product with xi_alpha (Source Tangent)
            # This converts the vector g_h into the scalar resolved stress contribution
            dot_val = 0.0
            for d in range(3):
                dot_val += g_h[d] * xi_alpha[d]

            local_val += rho * dot_val

        results[t_idx] = -local_val  # Apply negative sign as per original simple code

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

        # Compute GND and Local Tangent
        GND_map, xi_local_map = compute_gnd_and_xi_single_plane(data['density_map'], theta_arr)

        # Filter by cutoff density (Source Filtering)
        rows, cols = np.where(GND_map > min_cut_off_density)

        if len(rows) == 0: continue

        # Basis vectors for rotation
        if X.shape[0] > 1 and X.shape[1] > 1:
            u_vec = np.array([X[1, 0] - X[0, 0], Y[1, 0] - Y[0, 0], Z[1, 0] - Z[0, 0]])
            u_vec /= (np.linalg.norm(u_vec) + 1e-12)
            v_vec = np.array([X[0, 1] - X[0, 0], Y[0, 1] - Y[0, 0], Z[0, 1] - Z[0, 0]])
            v_vec /= (np.linalg.norm(v_vec) + 1e-12)
        else:
            continue

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

        # Normalize
        norms = np.linalg.norm(glob_xi, axis=1)
        valid_norm = norms > 1e-12
        glob_xi[valid_norm] = glob_xi[valid_norm] / norms[valid_norm, None]

        # Append to lists
        source_coords_list.append(positions)
        source_dens_list.append(densities)
        # Normal is constant for the whole mesh
        normals = np.tile(n_vec_global, (len(densities), 1))
        source_norm_list.append(normals)
        source_tan_list.append(glob_xi)

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

        # Identify valid grid points
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

            # CHANGE 3: Stress field is now 2D (Scalar field)
            stress_field = np.zeros((nx, ny))

            # Assign using advanced indexing
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