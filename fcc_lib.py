# fcc_lib.py (originally named slip_plane_functions.py)
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import numpy as np


def generate_grid(b, d, box_edge_length):
    """
    Generates a grid of points based on primitive vectors within a specified box.

    This function sets up a non-orthogonal coordinate system based on primitive
    vectors and then generates a grid of points. It determines the necessary
    range of integer indices (i, j, k) to completely cover a cubic simulation
    box and then returns the coordinates of all points within that box.

    Args:
        b (float): The magnitude of the Burgers vector (physical length unit).
        d (float): A scaling factor (multiplier) for the grid spacing relative to b.
                   (e.g., if d=500, the spacing is 500*b).
        box_edge_length (float): The side length of the cubic simulation box (L).

    Returns:
        tuple: A tuple containing:
            - XX (np.ndarray): 3D array of X coordinates for all grid points.
            - YY (np.ndarray): 3D array of Y coordinates for all grid points.
            - ZZ (np.ndarray): 3D array of Z coordinates for all grid points.
            - box_mask (np.ndarray): A boolean mask of the same shape as XX, YY, ZZ,
                                     where True indicates a point is inside the box.
    """
    L = box_edge_length

    # Define primitive unit vectors based on geometry
    u1 = np.array([1, 0, 0])
    u2 = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3), 0])
    u3 = np.array([1 / 2, np.sqrt(3) / 6, np.sqrt(2 / 3)])

    # Scale the unit vectors by the Burgers vector 'b' (physical size)
    # and the grid spacing multiplier 'd'
    p1, p2, p3 = (d) * u1, (d) * u2, (d) * u3

    # To find the range of indices (i, j, k) needed, we transform the
    # corners of the Cartesian box into the (p1, p2, p3) coordinate system.
    M = np.array([p1, p2, p3]).T
    M_inv = np.linalg.inv(M)
    corners = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L], [L, L, 0], [L, 0, L], [0, L, L], [L, L, L]])
    corners_ijk = M_inv @ corners.T

    # Determine the min and max integer indices that bound the box
    i_range = np.arange(int(np.floor(corners_ijk[0].min())), int(np.ceil(corners_ijk[0].max())) + 1)
    j_range = np.arange(int(np.floor(corners_ijk[1].min())), int(np.ceil(corners_ijk[1].max())) + 1)
    k_range = np.arange(int(np.floor(corners_ijk[2].min())), int(np.ceil(corners_ijk[2].max())) + 1)

    # Generate a grid of indices (I, J, K)
    I, J, K = np.meshgrid(i_range, j_range, k_range, indexing='ij')

    # Convert the index grid back to Cartesian coordinates
    XX = I * p1[0] + J * p2[0] + K * p3[0]
    YY = I * p1[1] + J * p2[1] + K * p3[1]
    ZZ = I * p1[2] + J * p2[2] + K * p3[2]

    # Create a boolean mask for points that fall within the Cartesian box bounds
    box_mask = (XX >= 0) & (XX < L) & (YY >= 0) & (YY < L) & (ZZ >= 0) & (ZZ < L)

    print(f"Grid generated. Total points inside box: {np.sum(box_mask)}")
    return XX, YY, ZZ, box_mask

def calculate_tetrahedron_slip_systems(b):
    """
    Calculates the slip systems based on a regular tetrahedron geometry.

    Args:
        b (float): Lattice constant (edge length 'a').

    Returns:
        np.ndarray: Array of shape (4, 3, 2, 3).
                    - 4: Unique Planes
                    - 3: Slip Directions per plane
                    - 2: Vector type (0=Normal, 1=Burgers/Direction)
                    - 3: Coordinates (x, y, z)
    """
    a = b

    # 1. Define the four vertices of the tetrahedron
    V = np.array([
        [0, 0, 0],  # V0
        [a, 0, 0],  # V1
        [a / 2, a * np.sqrt(3) / 2, 0],  # V2
        [a / 2, a * np.sqrt(3) / 6, a * np.sqrt(2 / 3)]  # V3
    ])

    # 2. Define the topology: Which vertices make up which face?
    # This corresponds to the logic in your original cross products.
    # Face 0: V0, V1, V2 (Bottom)
    # Face 1: V0, V1, V3 (Side)
    # Face 2: V0, V2, V3 (Side)
    # Face 3: V1, V2, V3 (Back)
    faces_indices = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ]

    # Initialize the output array
    # (4 planes, 3 directions, 2 vectors, 3 coords)
    slip_systems = np.zeros((4, 3, 2, 3))

    for i, indices in enumerate(faces_indices):
        # Get the 3 vertices for this face
        v_a = V[indices[0]]
        v_b = V[indices[1]]
        v_c = V[indices[2]]

        # --- A. Calculate Normal Vector (n) ---
        # We use the cross product of two edges.
        # (Replicating your specific cross product logic per face index)
        if i == 0:
            raw_n = np.cross(V[1] - V[0], V[2] - V[0])
        elif i == 1:
            raw_n = np.cross(V[1] - V[0], V[3] - V[0])
        elif i == 2:
            raw_n = np.cross(V[2] - V[0], V[3] - V[0])
        elif i == 3:
            raw_n = np.cross(V[2] - V[1], V[3] - V[1])

        # Normalize n
        n = raw_n / np.linalg.norm(raw_n)

        # --- B. Calculate Burgers Vectors (b) ---
        # The slip directions are the 3 edges defining the face triangle.
        # Edge 1: v_b -> v_a
        # Edge 2: v_c -> v_b
        # Edge 3: v_a -> v_c
        raw_directions = [
            v_b - v_a,
            v_c - v_b,
            v_a - v_c
        ]

        for j, raw_b in enumerate(raw_directions):
            # Normalize b
            b_vec = raw_b / np.linalg.norm(raw_b)

            # --- C. Populate Array ---
            # Dimension 0: Plane index i
            # Dimension 1: Direction index j
            # Dimension 2: 0 for Normal, 1 for Direction
            slip_systems[i, j, 0, :] = n
            slip_systems[i, j, 1, :] = b_vec

    return slip_systems


def calculate_plane_ids(xx, yy, zz, normals):
    """
    Calculates sorted, sequential plane IDs.

    CORRECTION: We normalize by the average grid spacing magnitude to ensure
    np.round() doesn't collapse all microscopic values to zero.
    """
    nx, ny, nz = xx.shape
    num_normals = len(normals)
    plane_ids_storage = np.zeros((nx, ny, nz, num_normals), dtype=np.int32)

    # Estimate a scaling factor based on the first non-zero coordinate found
    # This brings coordinates from 1e-10 range up to integer-like range (1, 2, 3...)
    # We use the max value in XX to estimate scale if available, else 1.0
    scale_factor = 1.0
    if np.max(xx) > 0:
        scale_factor = 1.0 / (np.max(xx) / nx)
        # Alternatively, since we know 'b' in main, we could pass it,
        # but estimating it from grid density is robust enough.
        # A safer generic approach for atomic scales:
        scale_factor = 1e9  # Convert Angstroms/Nanometers to larger numbers

    print("Calculating Plane IDs...")

    for idx, normal in enumerate(normals):
        # 1. Dot Product
        dot_product = xx * normal[0] + yy * normal[1] + zz * normal[2]

        # 2. Scaling & Rounding (THE FIX)
        # We multiply by a large number (1e10) so that 3.61e-10 becomes 3.61.
        # Then we round. This preserves the distinct planes.
        dot_product_scaled = dot_product * 1e10
        dot_product_rounded = np.round(dot_product_scaled, decimals=1)

        # 3. Find Unique Values
        unique_values = np.unique(dot_product_rounded)

        # 4. Assign IDs
        plane_ids_flat = np.searchsorted(unique_values, dot_product_rounded.ravel())
        plane_ids_grid = plane_ids_flat.reshape((nx, ny, nz))

        plane_ids_storage[..., idx] = plane_ids_grid

        print(f"  Normal {idx}: Found {len(unique_values)} unique planes.")

    return plane_ids_storage

def visualize_planes(XX, YY, ZZ, box_mask, slip_plane_ids, target_normal_idx):
    """
    Creates a 3D scatter plot to visualize specific slip planes.

    It automatically finds planes that exist within the simulation box and
    plots a sequence of them from the middle of the dataset to ensure they are visible.

    Args:
        XX (np.ndarray): 3D array of X coordinates of all grid points.
        YY (np.ndarray): 3D array of Y coordinates of all grid points.
        ZZ (np.ndarray): 3D array of Z coordinates of all grid points.
        box_mask (np.ndarray): Boolean mask for points inside the simulation box.
        slip_plane_ids (np.ndarray): 4D array of plane IDs from calculate_plane_ids.
        target_normal_idx (int): The index of the normal (0-3) to visualize.
    """
    # Get the plane IDs corresponding to the chosen normal
    ids_for_normal = slip_plane_ids[..., target_normal_idx]

    # 1. Filter to get IDs for points that are actually INSIDE the box.
    valid_ids_in_box = ids_for_normal[box_mask]

    if len(valid_ids_in_box) == 0:
        print("Error: No points found inside the box mask. Cannot visualize.")
        return

    # 2. Find the median ID to ensure we pick a plane that is centrally located.
    unique_ids_in_box = np.unique(valid_ids_in_box)
    middle_index = len(unique_ids_in_box) // 2

    # Pick 3 sequential IDs starting from the middle one for visualization.
    # This robustly selects planes that are guaranteed to exist and be visible.
    if middle_index + 2 < len(unique_ids_in_box):
        center_id = unique_ids_in_box[middle_index]
        target_ids = [center_id, center_id + 1, center_id + 2]
    elif len(unique_ids_in_box) >= 3:
        # If we are near the end of the ID list, pick the last 3
        target_ids = unique_ids_in_box[-3:]
    else:
        # If there are fewer than 3 planes in total, just plot what's available
        target_ids = unique_ids_in_box

    print(f"\nVisualizing Normal {target_normal_idx}")
    print(f"Available IDs in box range from {unique_ids_in_box.min()} to {unique_ids_in_box.max()}")
    print(f"Plotting sequential IDs: {target_ids}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # For efficiency, flatten the coordinate arrays and apply the box mask once.
    xx_flat_box = XX[box_mask]
    yy_flat_box = YY[box_mask]
    zz_flat_box = ZZ[box_mask]

    # Plot a faint background of all points in the box (subsampled for performance)
    ax.scatter(xx_flat_box[::20], yy_flat_box[::20], zz_flat_box[::20],
               s=1, alpha=0.05, c='gray', label='All Nodes (subsampled)')

    colors = ['red', 'blue', 'green']

    for i, pid in enumerate(target_ids):
        # Create a boolean mask for the current plane ID. This mask is applied to
        # `valid_ids_in_box`, which is already filtered to be inside the simulation box.
        mask_plane = (valid_ids_in_box == pid)

        if np.any(mask_plane):
            # Apply the plane mask to the flattened, box-filtered coordinates
            ax.scatter(xx_flat_box[mask_plane], yy_flat_box[mask_plane], zz_flat_box[mask_plane],
                       s=50, c=colors[i % 3], label=f'Plane ID {pid}')
        else:
            # This case is unlikely given how target_ids are chosen, but it is good practice to include.
            print(f"Warning: Plane ID {pid} has no points inside the box.")

    ax.set_title(f"Slip Plane Visualization for Normal {target_normal_idx}\nPlanes {target_ids}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


    ################################


import numpy as np


def structured_mesh_from_fcc_plane(points, tolerance=1e-15):
    """
    Takes a list of 3D points lying on an FCC plane (hexagonal lattice)
    and maps them to structured 2D arrays (X, Y, Z).

    Enforces a 120-degree angle between basis vectors a1 and a2.

    Returns:
        grid_X, grid_Y, grid_Z, a1, a2
    """
    points = np.array(points)
    n_points = points.shape[0]

    if n_points < 3:
        raise ValueError("Need at least 3 points to define a plane and basis.")

    # --- Step 1: Define the Origin ---
    origin = points[0]

    # --- Step 2: Find Basis Vectors ---
    diffs = points - origin
    dists = np.linalg.norm(diffs, axis=1)

    # Filter out the origin itself
    valid_indices = np.where(dists > tolerance)[0]

    if len(valid_indices) == 0:
        raise ValueError("All points are the same.")

    # Find the distance to the nearest neighbor
    min_dist = np.min(dists[valid_indices])

    # Get all points that are approximately at 'min_dist' distance
    neighbor_indices = valid_indices[np.abs(dists[valid_indices] - min_dist) < tolerance]

    if len(neighbor_indices) < 2:
        raise ValueError("Could not find enough neighbors to form a basis.")

    # Pick the first basis vector
    idx_a1 = neighbor_indices[0]
    a1 = points[idx_a1] - origin

    # Pick the second basis vector
    a2 = None
    for idx in neighbor_indices:
        candidate = points[idx] - origin
        # Check cross product to ensure they are not collinear
        cross_prod = np.cross(a1, candidate)
        if np.linalg.norm(cross_prod) > tolerance:
            a2 = candidate
            break

    if a2 is None:
        raise ValueError("Could not find a second independent basis vector.")

    # --- Step 2.5: Enforce 120 Degree Angle ---

    # Calculate current angle
    unit_a1 = a1 / np.linalg.norm(a1)
    unit_a2 = a2 / np.linalg.norm(a2)
    dot_product = np.dot(unit_a1, unit_a2)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    print(f"Initial angle found: {angle_deg:.2f} degrees")

    # If angle is approx 60 degrees (allowing for small float errors), flip a2
    if abs(angle_deg - 60.0) < 5.0:
        print("Angle is ~60 degrees. Flipping a2 to achieve 120 degrees.")
        a2 = -a2
        # Recalculate angle for verification
        unit_a2 = a2 / np.linalg.norm(a2)
        dot_product = np.dot(unit_a1, unit_a2)
        angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

    print(f"Final Basis used:\n a1: {a1}\n a2: {a2}")
    print(f"Final Angle: {angle_deg:.2f} degrees")

    # --- Step 3: Solve for Lattice Coordinates (u, v) ---
    # We use the (possibly flipped) a1 and a2 here
    Basis = np.column_stack((a1, a2))
    P_rel = (points - origin).T

    # Solve Basis * [u, v] = P_rel
    uv_coords, residuals, rank, s = np.linalg.lstsq(Basis, P_rel, rcond=None)
    uv_int = np.round(uv_coords).astype(int)

    # Sanity check reconstruction
    reconstructed = (Basis @ uv_coords).T + origin
    error = np.linalg.norm(points - reconstructed)
    if error > tolerance * n_points:
        print(f"Warning: High reconstruction error ({error}).")

    u_vals = uv_int[0, :]
    v_vals = uv_int[1, :]

    # --- Step 4: Create the Structured Grid ---
    u_min, u_max = np.min(u_vals), np.max(u_vals)
    v_min, v_max = np.min(v_vals), np.max(v_vals)

    dim_u = u_max - u_min + 1
    dim_v = v_max - v_min + 1

    print(f"Grid dimensions: {dim_u} x {dim_v}")

    grid_X = np.full((dim_u, dim_v), np.nan)
    grid_Y = np.full((dim_u, dim_v), np.nan)
    grid_Z = np.full((dim_u, dim_v), np.nan)

    for k in range(n_points):
        i = u_vals[k] - u_min
        j = v_vals[k] - v_min
        grid_X[i, j] = points[k, 0]
        grid_Y[i, j] = points[k, 1]
        grid_Z[i, j] = points[k, 2]

    # Return grids AND the corrected basis vectors
    return grid_X, grid_Y, grid_Z, a1, a2


def get_fcc_slip_systems():
    """
    Generates the standard FCC slip systems in a structured array.
    Shape: (4 planes, 3 directions, 2 vectors, 3 coords)
    """
    # Define the 4 unique {111} plane normals
    normals = [
        np.array([1, 1, 1]),
        np.array([-1, 1, 1]),
        np.array([1, -1, 1]),
        np.array([1, 1, -1])
    ]

    # Define the 3 <110> directions for each plane
    # Note: dot product of n . b must be 0 (orthogonal)
    directions = [
        # For plane (1, 1, 1)
        [np.array([0, 1, -1]), np.array([1, 0, -1]), np.array([1, -1, 0])],
        # For plane (-1, 1, 1)
        [np.array([0, 1, -1]), np.array([1, 0, 1]), np.array([1, 1, 0])],
        # For plane (1, -1, 1)
        [np.array([0, 1, 1]), np.array([1, 0, -1]), np.array([1, 1, 0])],
        # For plane (1, 1, -1)
        [np.array([0, 1, 1]), np.array([1, 0, 1]), np.array([1, -1, 0])]
    ]

    # Initialize the master array (4 planes, 3 dirs, 2 vectors(n,b), 3 coords)
    systems_array = np.zeros((4, 3, 2, 3))

    for i in range(4):
        for j in range(3):
            systems_array[i, j, 0, :] = normals[i]  # Store Normal n
            systems_array[i, j, 1, :] = directions[i][j]  # Store Direction b

    return systems_array


def calculate_rss_and_activity(loading_dir, load_val, crss, slip_systems):
    """
    Calculates RSS and Activity for FCC slip systems.

    Args:
        loading_dir (list): [u, v, w] loading direction.
        load_val (float): Magnitude of applied stress (MPa).
        crss (float): Critical Resolved Shear Stress (MPa).
        slip_systems (np.array): Array of shape (4, 3, 2, 3).

    Returns:
        rss_matrix (np.array): Shape (4, 3) containing RSS values.
        activity_matrix (np.array): Shape (4, 3) containing 1 (active) or 0 (inactive).
    """

    # 1. Normalize the Loading Direction
    L = np.array(loading_dir, dtype=float)
    L = L / np.linalg.norm(L)

    # 2. Initialize Output Arrays (4 planes, 3 directions)
    rss_matrix = np.zeros((4, 3))
    activity_matrix = np.zeros((4, 3), dtype=int)

    # 3. Iterate through the structure
    # i = plane index (0 to 3), j = direction index (0 to 2)
    for i in range(4):
        for j in range(3):
            # Extract Normal (n) and Direction (b)
            n_vec = slip_systems[i, j, 0, :]
            b_vec = slip_systems[i, j, 1, :]

            # Normalize n and b (Crucial for correct angles)
            n = n_vec / np.linalg.norm(n_vec)
            b = b_vec / np.linalg.norm(b_vec)

            # Calculate Schmid Factor: cos(phi) * cos(lambda)
            # cos(phi) = L . n
            # cos(lambda) = L . b
            schmid_factor = np.dot(L, n) * np.dot(L, b)

            # Calculate RSS
            rss = load_val * schmid_factor
            rss_matrix[i, j] = rss

            # Determine Activity
            # Slip is active if absolute RSS >= CRSS
            if abs(rss) >= crss:
                activity_matrix[i, j] = 1
            else:
                activity_matrix[i, j] = 0

    return rss_matrix, activity_matrix


import numpy as np


def reconstruct_active_slip_planes(active_s, node_on_slip_sys, XX, YY, ZZ):
    """
    Reconstructs specific slip plane instances based on active slip systems.

    Args:
        active_s (list or np.ndarray): List of active systems [[plane_type, dir_id], ...].
        node_on_slip_sys (dict): Mapping {(plane_type, stack_id): [[i, j, k], ...]}.
        XX, YY, ZZ (np.ndarray): 3D coordinate arrays.

    Returns:
        list: A list of dictionaries containing grid data and specific IDs.
    """

    # 1. Create a map of Active Directions for each Plane Normal Type (0-3)
    # Structure: { 0: {1, 2}, 1: {0}, ... }
    active_map = {}
    for sys in active_s:
        p_type = int(sys[0])
        d_id = int(sys[1])
        active_map.setdefault(p_type, set()).add(d_id)

    reconstructed_nodes = []

    # 2. Iterate through every specific plane instance found in the dictionary
    # Key format from your code: (plane_type, stack_id)
    for (p_type, p_stack_id), indices_list in node_on_slip_sys.items():

        # Only reconstruct if this plane type is in the active list
        if p_type in active_map:

            # Convert list of indices to numpy array for vectorization
            ids = np.array(indices_list)

            if ids.shape[0] == 0:
                continue

            # --- Vectorized Coordinate Extraction ---
            # Extract u, v, w indices columns
            u, v, w = ids[:, 0], ids[:, 1], ids[:, 2]

            # Create (N, 3) points array
            points = np.column_stack((XX[u, v, w], YY[u, v, w], ZZ[u, v, w]))

            # --- Generate Mesh ---
            try:
                # FIX APPLIED HERE:
                # Changed tolerance from 1e-5 to 1e-12.
                # Since your b is ~3e-9, points are separated by ~1e-9.
                # A tolerance of 1e-5 makes them look like the same point.
                # A tolerance of 1e-12 allows the code to distinguish them.
                x_grid, y_grid, z_grid, a_1, a_2 = structured_mesh_from_fcc_plane(points, tolerance=1e-14)

                # Construct the result dictionary with the requested IDs
                plane_data = {
                    "x_grid": x_grid,
                    "y_grid": y_grid,
                    "z_grid": z_grid,
                    "a_1": a_1,
                    "a_2": a_2,
                    "slip_plane_normal_id": p_type,  # 0, 1, 2, or 3
                    "slip_plane_stack_id": p_stack_id,  # The ID in the stack
                    "active_slip_system_ids": list(active_map[p_type])  # List of active directions
                }

                reconstructed_nodes.append(plane_data)

            except Exception as e:
                print(f"Skipping plane ({p_type}, {p_stack_id}) due to mesh error: {e}")

    return reconstructed_nodes




def generate_dislocation_loops_per_system(active_s, node_on_slip_sys, nodes_active, XX, YY, ZZ, b_vecs,
                                          target_density, min_size_m, max_size_m, b, box_length_m):
    """
    Generates loops ONLY on slip planes that have successfully generated meshes.
    """
    generated_loops = []

    # 1. Identify Valid Stacks from nodes_active
    # We create a set of (normal_id, stack_id) that actually have meshes.
    valid_stacks_set = set()
    for node in nodes_active:
        n_id = int(node['slip_plane_normal_id'])
        s_id = int(node['slip_plane_stack_id'])
        valid_stacks_set.add((n_id, s_id))

    real_volume_m3 = box_length_m ** 3
    target_length_per_system_m = target_density * real_volume_m3

    print(f"--- Generating Loops (Target Length: {target_length_per_system_m:.2e} m) ---")

    # 2. Iterate through all 12 slip systems
    for n_idx in range(4):
        for d_idx in range(3):
            if active_s[n_idx, d_idx] == 1:

                # Get all potential stacks for this normal
                potential_stacks = []
                # Check keys in node_on_slip_sys matching this normal
                for (k_n, k_s) in node_on_slip_sys.keys():
                    if k_n == n_idx:
                        potential_stacks.append(k_s)

                potential_stacks = list(set(potential_stacks))

                # FILTER: Only keep stacks that exist in valid_stacks_set
                available_stacks = [s for s in potential_stacks if (n_idx, s) in valid_stacks_set]

                if not available_stacks:
                    print(f"  > System ({n_idx}, {d_idx}) Active, but no valid meshes found. Skipping.")
                    continue

                current_system_length_m = 0.0
                normal_vec = b_vecs[n_idx, d_idx, 0, :]
                burgers_vec = b_vecs[n_idx, d_idx, 1, :]
                normal_vec = normal_vec / np.linalg.norm(normal_vec)

                # 3. Generate Loops
                while current_system_length_m < target_length_per_system_m:
                    stack_id = random.choice(available_stacks)
                    indices = node_on_slip_sys.get((n_idx, stack_id))

                    rand_pt_idx = random.choice(indices)
                    i, j, k = rand_pt_idx
                    center_point = np.array([XX[i, j, k], YY[i, j, k], ZZ[i, j, k]])

                    # Local coords
                    if np.abs(normal_vec[2]) < 0.9:
                        helper = np.array([0, 0, 1])
                    else:
                        helper = np.array([1, 0, 0])

                    u_vec = np.cross(normal_vec, helper)
                    u_vec /= np.linalg.norm(u_vec)
                    v_vec = np.cross(normal_vec, u_vec)
                    v_vec /= np.linalg.norm(v_vec)

                    # Size and Rotation
                    a_m = random.uniform(min_size_m, max_size_m)
                    c_m = random.uniform(min_size_m, max_size_m)
                    rotation_angle = random.uniform(0, 2 * np.pi)

                    theta = np.linspace(0, 2 * np.pi, 61)
                    x_local = a_m * np.cos(theta)
                    y_local = c_m * np.sin(theta)

                    x_rot = x_local * np.cos(rotation_angle) - y_local * np.sin(rotation_angle)
                    y_rot = x_local * np.sin(rotation_angle) + y_local * np.cos(rotation_angle)

                    loop_segments = (center_point + np.outer(x_rot, u_vec) + np.outer(y_rot, v_vec))

                    # Length calculation
                    segment_vectors = loop_segments[1:] - loop_segments[:-1]
                    actual_loop_length = np.sum(np.linalg.norm(segment_vectors, axis=1))
                    current_system_length_m += actual_loop_length

                    loop_data = {
                        "normal_id": int(n_idx),
                        "slip_system_id": int(d_idx),
                        "stack_id": int(stack_id),
                        "segments": loop_segments,
                        "burgers_vector": burgers_vec
                    }
                    generated_loops.append(loop_data)

                print(f"  > System ({n_idx}, {d_idx}) Generated {current_system_length_m:.2e} m")

    return generated_loops

#############################################



def compute_dislocation_density_maps(generated_loops, nodes_active, b,
                                     std_dev_mult=600, cutoff_mult=5):
    """
    Smears discrete dislocation loops onto the active slip plane meshes using
    a Gaussian distribution.
    """
    print("\n--- Computing Gaussian Smearing for Dislocation Density ---")

    sigma = std_dev_mult * b
    cutoff_radius = cutoff_mult * sigma
    two_sigma_sq = 2 * sigma ** 2
    # Normalization factor: ensures the integral of density equals the line length
    norm_factor = 1.0 / (2 * np.pi * sigma ** 2)

    # --- 1. Pre-build KDTrees for every active mesh ---
    mesh_acceleration_structures = {}

    print(f"Building search trees for {len(nodes_active)} active meshes...")

    for mesh_idx, mesh_data in enumerate(nodes_active):
        # Force cast to standard python int to avoid numpy int32/int64 mismatches
        n_id = int(mesh_data['slip_plane_normal_id'])
        s_id = int(mesh_data['slip_plane_stack_id'])

        X = mesh_data['x_grid']
        Y = mesh_data['y_grid']
        Z = mesh_data['z_grid']

        # Only build tree on valid points (not NaNs)
        valid_mask = ~np.isnan(X)
        valid_indices = np.argwhere(valid_mask)  # (row, col) indices

        valid_coords = np.column_stack((
            X[valid_mask],
            Y[valid_mask],
            Z[valid_mask]
        ))

        if len(valid_coords) == 0:
            continue

        tree = cKDTree(valid_coords)

        mesh_acceleration_structures[(n_id, s_id)] = {
            'tree': tree,
            'indices_map': valid_indices,
            'shape': X.shape
        }

    # --- 2. Iterate over loops and compute density ---
    QQ = {}
    missing_mesh_count = 0

    for loop_idx, loop in enumerate(generated_loops):
        n_id = int(loop['normal_id'])
        s_id = int(loop['stack_id'])

        # Check if we have a mesh for this loop's plane
        if (n_id, s_id) not in mesh_acceleration_structures:
            missing_mesh_count += 1
            continue

        struct = mesh_acceleration_structures[(n_id, s_id)]
        tree = struct['tree']
        indices_map = struct['indices_map']
        grid_shape = struct['shape']

        # Initialize Density Map
        density_map = np.zeros(grid_shape)

        points = loop['segments']

        # Calculate segment vectors, lengths, and centers
        # Vector from point i to i+1
        vecs = points[1:] - points[:-1]
        lengths = np.linalg.norm(vecs, axis=1)
        centers = (points[1:] + points[:-1]) / 2.0

        # --- 3. Smear each segment ---
        for i, center in enumerate(centers):
            seg_len = lengths[i]

            # Find mesh nodes within cutoff radius
            neighbor_indices = tree.query_ball_point(center, cutoff_radius)

            if not neighbor_indices:
                continue

            # Get coordinates of these neighbors
            neighbor_coords = tree.data[neighbor_indices]

            # Calculate Gaussian weights
            diff = neighbor_coords - center
            dist_sq = np.sum(diff ** 2, axis=1)

            # Weight = Length * Gaussian_Value
            weights = (seg_len * norm_factor) * np.exp(-dist_sq / two_sigma_sq)

            # Map back to 2D grid
            for k, tree_idx in enumerate(neighbor_indices):
                r, c = indices_map[tree_idx]
                density_map[r, c] += weights[k]

        QQ[loop_idx] = {
            "density_map": density_map,
            "burgers_vector": loop['burgers_vector'],
            "normal_id": n_id,
            "stack_id": s_id,
            "slip_system_id": int(loop['slip_system_id'])
        }

    if missing_mesh_count > 0:
        print(f"Warning: {missing_mesh_count} loops were skipped because their slip plane mesh was missing.")
        print("Ensure 'generate_dislocation_loops_per_system' only picks stacks present in 'nodes_active'.")

    print(f"Density mapping complete. Generated maps for {len(QQ)} loops.")
    return QQ