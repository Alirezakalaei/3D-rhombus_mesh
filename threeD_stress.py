import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


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
# NEW: GND AND TANGENT COMPUTATION
# =============================================================================

@njit(fastmath=True)
def compute_gnd_and_xi_single_plane(density_map, theta_array):
    """
    Compute GND density and tangent vector xi for a SINGLE plane/stack.

    Args:
        density_map: (nx, ny, n_theta) - The angular density distribution
        theta_array: (n_theta) - The angles corresponding to the bins

    Returns:
        GND: (nx, ny) - Scalar dislocation density magnitude
        xi: (nx, ny, 3) - Normalized tangent vector in LOCAL coordinates (2D plane)
    """
    nx, ny, n_thet = density_map.shape
    GND = np.zeros((nx, ny))
    xi_local = np.zeros((nx, ny, 3))  # 3D vector, but z component is 0 locally

    cos_theta = np.cos(theta_array)
    sin_theta = np.sin(theta_array)

    for i in range(nx):
        for j in range(ny):
            # Vector sum of density contributions
            vec_x = 0.0
            vec_y = 0.0

            for k in range(n_thet):
                rho = density_map[i, j, k]
                vec_x += rho * cos_theta[k]
                vec_y += rho * sin_theta[k]

            # Magnitude is the GND density
            mag = np.sqrt(vec_x ** 2 + vec_y ** 2)
            GND[i, j] = mag

            # Direction is the tangent vector xi
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
# MURA KERNEL (UPDATED FOR TANGENT XI)
# =============================================================================

@njit(fastmath=True)
def mura_g_tensor(r_vec, n_alpha, xi_alpha, b_prime, C, mu, nu, a, epsilon_tens, delta):
    """
    Computes the interaction vector g_h based on Mura's equation.

    CRITICAL CHANGE: 
    In the original formula, the source term usually involves (b x xi). 
    However, your previous code used b_alpha. 
    If your model assumes the source is a loop where b is constant, 
    the 'vector' nature comes from the tangent line xi.

    Standard Mura formula often uses: epsilon_jsh * xi_s * ...

    Here, we replace the source vector component with the calculated xi_alpha.
    """
    # Compute R terms with regularization parameter 'a'
    R2 = r_vec[0] ** 2 + r_vec[1] ** 2 + r_vec[2] ** 2 + a ** 2
    R = np.sqrt(R2)
    invR3 = 1.0 / (R ** 3)
    invR5 = 1.0 / (R ** 5)

    prefactor_base = 1.0 / (16.0 * np.pi * mu * (1.0 - nu))
    g_h = np.zeros(3)

    for h in range(3):
        total = 0.0
        for p in range(3):
            np_p = n_alpha[p]
            for q in range(3):
                # In Mura's formula for loops, this term often relates to the tangent.
                # Assuming xi_alpha takes the place of the Burgers vector direction 
                # in the contraction or interacts with it.
                # Based on standard dislocation theory, the source term is epsilon_imn * xi_m * b_n.
                # However, sticking to your provided tensor structure:
                # We use xi_alpha[q] as the source vector component.
                vec_q = xi_alpha[q]

                for r_idx in range(3):
                    for s in range(3):
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
                                        delta_rk = delta[r_idx, k]
                                        delta_rl = delta[r_idx, l]
                                        delta_kl = delta[k, l]

                                        term1 = 6.0 * delta_rk * a ** 2 * (1.0 - nu) * x_l * invR5
                                        term2 = (3.0 - 4.0 * nu) * delta_rk * x_l * invR3
                                        term3 = 3.0 * x_r * x_k * x_l * invR5
                                        term4 = - (delta_rl * x_k + delta_kl * x_r) * invR3

                                        bracket = term1 + term2 + term3 + term4
                                        contrib = coeff * eps_jsh * bi_prime * Cijkl * bracket
                                        total += contrib
        g_h[h] = total
    return g_h


# =============================================================================
# MAIN COMPUTATION FUNCTION (UPDATED)
# =============================================================================

def compute_interaction_stress_mura(QQ_smeared,C_tensor, eps_tensor, delta_tensor, nodes_active, b_vecs, b, mu, nu, a_param,
                                    cut_off_dist=3000, min_cut_off_density=1e9):
    """
    Computes Mura interaction stress using computed GND and Tangent vectors (xi).
    """
    print(f"\n--- Computing Mura Interaction Stress (a={a_param:.2e}, cutoff={cut_off_dist:.2e}) ---")


    # 2. Build Source Points (KDTree)
    source_coords = []
    source_props = []  # (GND_magnitude, normal_vec, xi_vec_global)

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

    # Pre-calculate Theta Array (0 to 2pi)
    # Assuming QQ_smeared has nthetaintervals implicit in shape
    first_key = list(QQ_smeared.keys())[0]
    n_theta = QQ_smeared[first_key]['density_map'].shape[2]
    theta_arr = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

    print("  > Extracting GND and Tangent Vectors for Sources...")
    count_sources = 0

    for key, data in QQ_smeared.items():
        n_id = data['normal_id']
        s_id = data['stack_id']
        n_vec_global = plane_normals[n_id]

        if (n_id, s_id) not in mesh_lookup: continue
        X, Y, Z = mesh_lookup[(n_id, s_id)]

        # --- NEW: Compute GND and Local Tangent (xi) ---
        # This replaces the simple summation
        GND_map, xi_local_map = compute_gnd_and_xi_single_plane(data['density_map'], theta_arr)

        # Filter by cutoff density
        rows, cols = np.where(GND_map > min_cut_off_density)

        # Need to rotate local xi (2D in plane) to global xi (3D)
        # We need a basis for the plane. 
        # Assuming X, Y, Z grids were generated such that local x aligns with some vector u, 
        # and local y with v.
        # Simple approach: If the mesh is regular, we can deduce basis from neighbors, 
        # or assume the plane generation logic.
        # For FCC planes, usually we have specific basis vectors.
        # Here, we approximate global xi by mapping the local 2D direction onto the 3D plane.

        # Find basis vectors of the plane from the grid itself (finite difference of grid)
        # Taking central point to avoid edge issues, or just 0,0 and 0,1
        if X.shape[0] > 1 and X.shape[1] > 1:
            # Vector along local X axis
            u_vec = np.array([X[1, 0] - X[0, 0], Y[1, 0] - Y[0, 0], Z[1, 0] - Z[0, 0]])
            u_vec /= np.linalg.norm(u_vec)
            # Vector along local Y axis
            v_vec = np.array([X[0, 1] - X[0, 0], Y[0, 1] - Y[0, 0], Z[0, 1] - Z[0, 0]])
            v_vec /= np.linalg.norm(v_vec)
        else:
            continue  # Skip single point meshes

        for r, c in zip(rows, cols):
            dens = GND_map[r, c]
            pos = np.array([X[r, c], Y[r, c], Z[r, c]])

            # Convert local xi (cos, sin) to global 3D vector
            local_xi = xi_local_map[r, c]  # [x, y, 0]
            global_xi = local_xi[0] * u_vec + local_xi[1] * v_vec

            # Normalize global_xi just in case
            norm_xi = np.linalg.norm(global_xi)
            if norm_xi > 1e-12:
                global_xi /= norm_xi

            source_coords.append(pos)
            # Store: (Density Magnitude, Plane Normal, Tangent Vector)
            source_props.append((dens, n_vec_global, global_xi))
            count_sources += 1

    if count_sources == 0:
        print("  ! No sources found above density cutoff.")
        return {}

    source_tree = cKDTree(np.array(source_coords))
    source_props_arr = source_props

    # 3. Compute Stress for Targets
    stress_results = {}

    print(f"  > Computing targets for {len(nodes_active)} active stacks...")

    for mesh_idx, mesh in enumerate(nodes_active):
        n_id_target = int(mesh['slip_plane_normal_id'])
        s_id_target = int(mesh['slip_plane_stack_id'])

        X_target = mesh['x_grid']
        Y_target = mesh['y_grid']
        Z_target = mesh['z_grid']

        # Identify valid grid points (geometry)
        valid_mask = ~np.isnan(X_target)
        target_indices = np.argwhere(valid_mask)

        if len(target_indices) == 0: continue

        # Loop over all 3 possible slip systems for this plane
        for sys_id in range(3):

            # Get the target Burgers vector b'
            b_prime = b_vecs[n_id_target, sys_id, 1, :]

            # Initialize Stress Array for this system
            nx, ny = X_target.shape
            stress_field = np.zeros((nx, ny, 3))  # Vector result g_h

            # Iterate over all valid nodes in the mesh
            for idx in target_indices:
                r, c = idx
                target_pos = np.array([X_target[r, c], Y_target[r, c], Z_target[r, c]])

                # Find sources within cut_off_dist
                neighbor_idxs = source_tree.query_ball_point(target_pos, cut_off_dist)

                if not neighbor_idxs:
                    continue

                local_g = np.zeros(3)

                for src_idx in neighbor_idxs:
                    src_pos = source_coords[src_idx]
                    # Unpack updated props: Density, Normal, Tangent(Xi)
                    rho_alpha, n_alpha, xi_alpha = source_props_arr[src_idx]

                    r_vec = target_pos - src_pos

                    dist_sq = np.dot(r_vec, r_vec)
                    if dist_sq < 1e-20: continue

                    # Compute Mura Kernel with Tangent Vector
                    g_val = mura_g_tensor(r_vec, n_alpha, xi_alpha, b_prime,
                                          C_tensor, mu, nu, a_param, eps_tensor, delta_tensor)

                    local_g += rho_alpha * g_val

                stress_field[r, c, :] = local_g

            # Save result
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

# ... (Include other previous functions like generate_grid, etc. here) ...