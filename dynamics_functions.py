import numpy as np


def compute_effective_stress(interaction_stress_data, load_direction, app_load, CRSS, b_vecs):
    """
    Computes effective stress at each point for every active system.
    Formula: Effective Stress = Internal Stress + Resolved Applied Stress + Peierls Stress (CRSS)

    Args:
        interaction_stress_data (dict): Output from compute_interaction_stress_mura
        load_direction (array): Vector [x, y, z]
        app_load (float): Magnitude of applied load (Pa)
        CRSS (float): Peierls stress / Critical Resolved Shear Stress (Pa)
        b_vecs (array): The slip system geometry array (4, 3, 2, 3)

    Returns:
        dict: Same structure as interaction_stress_data but with effective stress values.
    """
    print(f"\n--- Computing Effective Stress ---")

    effective_stress_data = {}

    # 1. Normalize the global load direction once
    L = np.array(load_direction, dtype=float)
    norm_L = np.linalg.norm(L)
    if norm_L > 1e-12:
        L = L / norm_L
    else:
        # Fallback if 0 vector provided
        L = np.array([0, 0, 1])

        # 2. Iterate over all computed internal stress fields
    for key, data in interaction_stress_data.items():
        # Unpack the key to identify the specific system
        # Key format: (normal_id, stack_id, slip_system_id)
        nid, sid, sysid = key

        # Get the internal stress map (2D array)
        sigma_internal = data['stress_field']

        # 3. Calculate Resolved Applied Stress for this specific system
        # Retrieve geometry from b_vecs
        # b_vecs structure: [plane_index, system_index, vector_type, xyz]
        # vector_type 0 is Normal, 1 is Burgers vector
        n_vec = b_vecs[nid, sysid, 0, :]
        b_vec = b_vecs[nid, sysid, 1, :]

        # Normalize n and b
        n_unit = n_vec / np.linalg.norm(n_vec)
        b_unit = b_vec / np.linalg.norm(b_vec)

        # Calculate Schmid Factor: (L . n) * (L . b)
        schmid_factor = np.dot(L, n_unit) * np.dot(L, b_unit)

        # Scalar value of resolved applied stress
        sigma_applied = app_load * schmid_factor

        # 4. Compute Effective Stress
        # Summation: Internal (Array) + Applied (Scalar) + Peierls (Scalar)
        # We perform this operation on the whole array.
        # Note: This will add stress to the "background" (zero-valued areas) as well.
        # If you strictly want to keep the background as 0, you would need to mask it
        # using the mesh validity, but mathematically this is the requested summation.
        sigma_effective = sigma_internal + sigma_applied + CRSS

        # 5. Save in the exact format of the input
        effective_stress_data[key] = {
            "stress_field": sigma_effective,
            "slip_plane_normal_id": nid,
            "slip_plane_stack_id": sid,
            "slip_system_id": sysid,
            # Optional: store the scalar components for reference if needed later
            "resolved_applied_val": sigma_applied,
            "peierls_val": CRSS
        }

    return effective_stress_data