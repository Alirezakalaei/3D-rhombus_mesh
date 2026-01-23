# fcc_lib.py (originally named slip_plane_functions.py)
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import numpy as np
from scipy.spatial.distance import cdist

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

def structured_mesh_from_fcc_plane(points, tolerance=1e-15):
    """
    Reconstructs a structured grid from a cloud of points on an FCC plane.

    Algorithm:
    1. Discard if nodes < 5.
    2. Find nearest neighbor distance (d).
    3. Find two neighbors forming a 120-degree angle to define basis vectors (a1, a2).
    4. Scan/Map all points to integer coordinates (n1, n2) using this basis.
    5. Create a grid from min(n) to max(n) and fill missing spots with NaN (Null).
    """

    num_points = len(points)

    # --- FILTER: Discard small stacks ---
    if num_points < 5:
        return None, None, None, None, None

    # --- Step 1: Determine Basis Vectors (a1, a2) ---
    a1 = None
    a2 = None

    # Calculate nearest neighbor distance (d)
    # We use a subset for speed, assuming homogeneous lattice
    subset = points[:min(50, num_points)]
    dists = cdist(subset, subset)
    valid_dists = dists[dists > 1e-8]  # Exclude self-distance (0)

    if len(valid_dists) == 0:
        return None, None, None, None, None

    nn_dist = np.min(valid_dists)

    # Search for basis vectors with 120 degree separation
    found_basis = False

    # Try the first few points as potential origins to find a valid basis
    for i in range(min(10, num_points)):
        p0 = points[i]

        # Find neighbors within tolerance of d
        dist_from_p0 = np.linalg.norm(points - p0, axis=1)
        # 5% tolerance for thermal noise or numerical float errors
        neighbor_indices = np.where((dist_from_p0 > nn_dist * 0.95) &
                                    (dist_from_p0 < nn_dist * 1.05))[0]

        if len(neighbor_indices) < 2:
            continue

        neighbors = points[neighbor_indices]
        vecs = neighbors - p0

        # Normalize for angle calculation
        norms = np.linalg.norm(vecs, axis=1)
        unit_vecs = vecs / norms[:, np.newaxis]

        # Check angles between pairs of neighbors
        for j in range(len(unit_vecs)):
            for k in range(j + 1, len(unit_vecs)):
                dot_prod = np.dot(unit_vecs[j], unit_vecs[k])

                # cos(120) = -0.5.
                # We check range [-0.6, -0.4] to be safe.
                if -0.6 < dot_prod < -0.4:
                    a1 = vecs[j]
                    a2 = vecs[k]
                    found_basis = True
                    break
            if found_basis: break
        if found_basis: break

    # Fallback: If 120 not found (e.g., at edges), try 60 degrees and flip vector
    if not found_basis:
        for i in range(min(10, num_points)):
            p0 = points[i]
            dist_from_p0 = np.linalg.norm(points - p0, axis=1)
            neighbor_indices = np.where((dist_from_p0 > nn_dist * 0.95) &
                                        (dist_from_p0 < nn_dist * 1.05))[0]
            if len(neighbor_indices) < 2: continue

            neighbors = points[neighbor_indices]
            vecs = neighbors - p0
            unit_vecs = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]

            for j in range(len(unit_vecs)):
                for k in range(j + 1, len(unit_vecs)):
                    dot_prod = np.dot(unit_vecs[j], unit_vecs[k])
                    # cos(60) = 0.5
                    if 0.4 < dot_prod < 0.6:
                        a1 = vecs[j]
                        a2 = vecs[k]
                        # Construct 120 basis from 60 basis: a2_new = a2_old - a1
                        a2 = a2 - a1
                        found_basis = True
                        break
                if found_basis: break
            if found_basis: break

    if not found_basis:
        return None, None, None, None, None

    # --- Step 2: Scan/Map points to integer indices (n1, n2) ---
    # We solve P = Origin + n1*a1 + n2*a2 for n1, n2

    M = np.column_stack((a1, a2))  # Matrix [a1, a2]
    origin = points[0]  # Arbitrary origin for the grid
    P_rel = (points - origin).T

    # Linear Least Squares to find indices
    # (M^T M)^-1 M^T * P_rel
    coeffs = np.linalg.inv(M.T @ M) @ M.T @ P_rel

    indices = np.round(coeffs).astype(int)
    n1_vals = indices[0, :]
    n2_vals = indices[1, :]

    # Validation: Check if points actually fall on grid nodes
    reconstructed_points = (M @ indices).T + origin
    errors = np.linalg.norm(points - reconstructed_points, axis=1)
    if np.max(errors) > nn_dist * 0.1:
        # Points do not fit this basis well
        return None, None, None, None, None

    # --- Step 3: Create Grid and Fill ---
    n1_min, n1_max = np.min(n1_vals), np.max(n1_vals)
    n2_min, n2_max = np.min(n2_vals), np.max(n2_vals)

    dim1 = n1_max - n1_min + 1
    dim2 = n2_max - n2_min + 1

    # Initialize with Null (NaN)
    x_grid = np.full((dim1, dim2), np.nan)
    y_grid = np.full((dim1, dim2), np.nan)
    z_grid = np.full((dim1, dim2), np.nan)

    # Map indices to 0-based array coordinates
    idx_1 = n1_vals - n1_min
    idx_2 = n2_vals - n2_min

    x_grid[idx_1, idx_2] = points[:, 0]
    y_grid[idx_1, idx_2] = points[:, 1]
    z_grid[idx_1, idx_2] = points[:, 2]

    return x_grid, y_grid, z_grid, a1, a2


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







def reconstruct_active_slip_planes(active_s, node_on_slip_sys, XX, YY, ZZ):
    """
    Reconstructs specific slip plane instances based on active slip systems.

    Args:
        active_s: List or Matrix of active systems.
        node_on_slip_sys: Dictionary {(plane_type, stack_id): [indices]}.
        XX, YY, ZZ: 3D coordinate arrays.

    Returns:
        List of dictionaries containing grid data and requested IDs.
    """

    # 1. Parse Active Systems into a Map
    # Structure: { plane_type_id: {dir_id_1, dir_id_2}, ... }
    active_map = {}

    # Handle numpy array input (common in previous steps)
    if isinstance(active_s, np.ndarray):
        rows, cols = active_s.shape
        for p_type in range(rows):
            for d_id in range(cols):
                if active_s[p_type, d_id] == 1:
                    active_map.setdefault(p_type, set()).add(d_id)
    # Handle list input [[plane, dir], ...]
    else:
        for sys in active_s:
            p_type = int(sys[0])
            d_id = int(sys[1])
            active_map.setdefault(p_type, set()).add(d_id)

    reconstructed_nodes = []

    # 2. Iterate through every specific plane instance found
    for (p_type, p_stack_id), indices_list in node_on_slip_sys.items():

        # Only process if this plane type is active
        if p_type in active_map:

            ids = np.array(indices_list)

            # Pre-check: strictly less than 5 nodes are ignored
            if ids.shape[0] < 5:
                continue

            # Extract 3D coordinates
            u, v, w = ids[:, 0], ids[:, 1], ids[:, 2]
            points = np.column_stack((XX[u, v, w], YY[u, v, w], ZZ[u, v, w]))

            try:
                # Generate Mesh using the specific algorithm
                x_grid, y_grid, z_grid, a_1, a_2 = structured_mesh_from_fcc_plane(points)

                # If None, it means mesh construction failed (e.g. no valid basis found)
                if x_grid is None:
                    continue

                # Construct the result dictionary with ALL requested IDs
                plane_data = {
                    "x_grid": x_grid,
                    "y_grid": y_grid,
                    "z_grid": z_grid,
                    "a_1": a_1,
                    "a_2": a_2,
                    # Requested Output IDs:
                    "slip_plane_normal_id": p_type,  # The Plane Type (0-3)
                    "slip_plane_stack_id": p_stack_id,  # The Stack ID
                    "active_slip_system_ids": list(active_map[p_type])  # List of active directions
                }

                reconstructed_nodes.append(plane_data)

            except Exception as e:
                print(f"Skipping plane ({p_type}, {p_stack_id}) due to error: {e}")

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



def compute_density_and_theta_maps(generated_loops, nodes_active, b,
                                   nthetaintervals, std_dev_mult=600, cutoff_mult=5):
    """
    Computes a combined Dislocation Density and Theta map.

    Instead of a simple 2D density map, this function produces a 3D array where the
    density is binned by the orientation angle (theta) of the dislocation line relative
    to its Burgers vector.

    Args:
        generated_loops: List of loop data dictionaries.
        nodes_active: List of mesh data dictionaries.
        b: Burgers vector magnitude.
        nthetaintervals (int): Number of intervals to divide the 0-2pi (0-360 deg) range.
        std_dev_mult: Multiplier for Gaussian sigma.
        cutoff_mult: Multiplier for cutoff radius.

    Returns:
        QQ (dict): Dictionary keyed by loop index containing:
            - "density_map": nd.array of shape (nx, ny, nthetaintervals)
            - "normal_id"
            - "stack_id"
            - "slip_system_id"
            - "burgers_vector"
    """
    print(f"\n--- Computing Density Maps with Theta Binning ({nthetaintervals} intervals) ---")

    sigma = std_dev_mult * b
    cutoff_radius = cutoff_mult * sigma
    two_sigma_sq = 2 * sigma ** 2
    # Normalization factor: ensures the integral of density equals the line length
    norm_factor = 1.0 / (2 * np.pi * sigma ** 2)

    # Define bin edges for 0 to 360 degrees
    # We use degrees for easier debugging, but math works same for radians
    bin_edges = np.linspace(0, 360, nthetaintervals + 1)

    # --- 1. Pre-build KDTrees for every active mesh ---
    mesh_acceleration_structures = {}

    print(f"Building search trees for {len(nodes_active)} active meshes...")

    for mesh_idx, mesh_data in enumerate(nodes_active):
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

    # --- 2. Iterate over loops ---
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
        grid_shape_2d = struct['shape']

        # Initialize 3D Density Map: (nx, ny, nthetaintervals)
        # We add the 3rd dimension for theta bins
        density_map_3d = np.zeros((grid_shape_2d[0], grid_shape_2d[1], nthetaintervals))

        points = loop['segments']
        b_vec = loop['burgers_vector']

        # Normalize Burgers vector
        b_norm = np.linalg.norm(b_vec)
        b_unit = b_vec / b_norm if b_norm > 0 else b_vec

        # Calculate segment vectors, lengths, and centers
        vecs = points[1:] - points[:-1]
        lengths = np.linalg.norm(vecs, axis=1)
        centers = (points[1:] + points[:-1]) / 2.0

        # --- 3. Process Segments ---
        for i, center in enumerate(centers):
            seg_len = lengths[i]
            if seg_len == 0: continue

            # --- A. Calculate Angle Theta ---
            t_unit = vecs[i] / seg_len

            # Dot product for angle: cos(theta) = t . b
            dot_product = np.dot(t_unit, b_unit)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)  # Result is 0 to 180

            # Since dislocation character is usually 0 (screw) to 90 (edge),
            # but mathematical angle is 0 to 180.
            # If you specifically need 0-360 orientation (direction matters),
            # we need a reference vector to determine sign.
            # For standard character angle, 0-180 is usually sufficient.
            # If you strictly need 0-2pi mapping, we assume the input request implies 0-360.
            # However, arccos only gives 0-180.
            # To get full 0-360, we'd need a plane normal cross product check.
            # Assuming simple 0-180 mapping is acceptable or 0-360 is mapped:

            # Map angle to bin index
            # np.digitize returns 1 for first bin, so we subtract 1 to get 0-based index
            bin_idx = np.digitize(angle_deg, bin_edges) - 1

            # Handle edge case where angle is exactly 360 (goes to index nthetaintervals)
            if bin_idx >= nthetaintervals:
                bin_idx = nthetaintervals - 1

            # --- B. Spatial Smearing ---
            # Find mesh nodes within cutoff radius
            neighbor_indices = tree.query_ball_point(center, cutoff_radius)

            if not neighbor_indices:
                continue

            neighbor_coords = tree.data[neighbor_indices]

            # Calculate Gaussian weights
            diff = neighbor_coords - center
            dist_sq = np.sum(diff ** 2, axis=1)

            # Weight = Length * Gaussian_Value
            weights = (seg_len * norm_factor) * np.exp(-dist_sq / two_sigma_sq)

            # --- C. Add to specific Theta Bin ---
            for k, tree_idx in enumerate(neighbor_indices):
                r, c = indices_map[tree_idx]
                # Add density ONLY to the layer corresponding to the angle bin
                density_map_3d[r, c, bin_idx] += weights[k]

        QQ[loop_idx] = {
            "density_map": density_map_3d,  # Now 3D
            "burgers_vector": loop['burgers_vector'],
            "normal_id": n_id,
            "stack_id": s_id,
            "slip_system_id": int(loop['slip_system_id'])
        }

    if missing_mesh_count > 0:
        print(f"Warning: {missing_mesh_count} loops were skipped because their slip plane mesh was missing.")

    print(f"Computation complete. Generated 3D maps for {len(QQ)} loops.")
    return QQ



def smear_configurational_density(QQ_input, nthetaintervals, std_dev_rad=(5 * np.pi / 180)):
    """
    Smears the dislocation density in theta-space (configurational space) using a
    Gaussian distribution. It processes the dictionary output from
    compute_density_and_theta_maps and handles multiple slip systems.

    Args:
        QQ_input (dict): The dictionary returned by compute_density_and_theta_maps.
                         Keys are loop indices, values contain 'density_map' (nx, ny, ntheta).
        nthetaintervals (int): Total number of theta bins (e.g., 360).
        std_dev_rad (float): Standard deviation for the Gaussian kernel in radians.
                             Default is 5 degrees (5 * pi / 180).

    Returns:
        dict: A new dictionary with the same structure as QQ_input, but with
              smeared density maps.
    """
    print(f"\n--- Smearing Density in Theta Space (std={std_dev_rad:.4f} rad) ---")

    # 1. Generate the Gaussian Kernel
    # Calculate angular step size
    d_theta = 2 * np.pi / nthetaintervals

    # Determine how many bins out to calculate (5 sigma covers >99.9%)
    n_dist = int(np.ceil(5 * std_dev_rad / d_theta))

    # Create an array of integer offsets [-n, ..., 0, ..., +n]
    kernel_indices = np.arange(-n_dist, n_dist + 1)

    # Convert offsets to physical angles for Gaussian calculation
    dist_vals = kernel_indices * d_theta

    # Calculate Gaussian coefficients
    kernel = (1 / (np.sqrt(2 * np.pi * std_dev_rad ** 2))) * np.exp(-(dist_vals ** 2) / (2 * std_dev_rad ** 2))

    # Normalize kernel so total probability sum is 1.0 (conserves density)
    kernel = kernel / np.sum(kernel)

    print(f"Kernel size: {len(kernel)} bins (spanning +/- {n_dist} bins)")

    QQ_smeared = {}

    # 2. Iterate through every loop/slip system in the input dictionary
    for key, data in QQ_input.items():

        # Extract the raw "sharp" map: shape (nx, ny, nthetaintervals)
        raw_map = data['density_map']

        # Initialize the new smeared map
        smeared_map = np.zeros_like(raw_map)

        # 3. Apply Circular Convolution along the Theta axis
        # We iterate through every 'source' theta bin. If there is density there,
        # we spread it to the 'target' neighbor bins based on the kernel.

        for t_source in range(nthetaintervals):
            # Optimization: Extract the 2D slice for this specific angle
            source_slice = raw_map[:, :, t_source]

            # If this angle has no dislocations, skip calculation
            if np.sum(source_slice) == 0:
                continue

            # Distribute this density to neighbors
            for k_idx, offset in enumerate(kernel_indices):
                weight = kernel[k_idx]

                # Calculate target bin index with circular wrapping (modulo)
                # e.g., if t_source is 359 and offset is +1, t_target becomes 0
                t_target = (t_source + offset) % nthetaintervals

                # Add weighted density to the target bin
                smeared_map[:, :, t_target] += source_slice * weight

        # 4. Store Data in Output Dictionary
        # Preserving all ID tags (Slip plane ID, Slip system ID, Stack ID)
        QQ_smeared[key] = {
            "density_map": smeared_map,
            "normal_id": data['normal_id'],  # Slip Plane ID (0-3)
            "stack_id": data['stack_id'],  # Stack ID
            "slip_system_id": data['slip_system_id'],  # Slip System ID (0-2)
            "burgers_vector": data['burgers_vector']
        }

    print("Smearing complete.")
    return QQ_smeared