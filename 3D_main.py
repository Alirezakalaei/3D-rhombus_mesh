
import numpy as np
import matplotlib

matplotlib.use('TKAgg')

import matplotlib.pyplot as plt

# Import the functions from our custom module.
# This assumes 'slip_plane_functions.py' is in the same directory.
from fcc_lib import (
    generate_grid,
    calculate_tetrahedron_slip_systems,
    calculate_plane_ids,
    visualize_planes,
    structured_mesh_from_fcc_plane,
    calculate_rss_and_activity,
    reconstruct_active_slip_planes,
    generate_dislocation_loops_per_system,
    compute_density_and_theta_maps,
    smear_configurational_density,
    normalize_density_to_target

)

from threeD_stress import (get_elastic_constants,compute_interaction_stress_mura, get_epsilon_delta)

# =============================================================================
# SCRIPT FOR INTERACTIVE ANALYSIS
#
# This version of the main script is designed to be run in an interactive
# environment like Spyder, IPython, or a Jupyter Notebook.
# By placing the code at the top level (not inside a main() function), all
# variables created (e.g., XX, YY, ZZ, slip_plane_ids) will remain in the
# global workspace after the script finishes. This allows for direct
# inspection and further analysis in the console.


# =============================================================================
# 1. SETUP PARAMETERS
# =============================================================================
print("--- 1. Setting up parameters ---")
b = 2.56*(10**-10)  # Lattice constant (e.g., for Copper in Angstroms)
target_density= 10**11
d = 500*b
CRSS = .5*10**6
app_load = 20*10**6
mu = 46e9 #Pa
nu = .34
a= 1*b # singularity removal
min_density_val = 1e5 #m^-2
cutoff_dist_val = 4000*b

load_direction= np.array([0, 0, 1])
b_vecs = calculate_tetrahedron_slip_systems(b)
box_edge_length = 20000 * b
n_grid = np.ceil(box_edge_length / d)
nthetaintervals = 360  # Example: Divide 0-360 degrees into 18 bins (20 degrees each)
# =============================================================================
# 2. EXECUTE COMPUTATIONAL STEPS
# =============================================================================
print("\n--- 2. Generating Grid ---")
# Generate the grid points and the mask for the simulation box
XX, YY, ZZ, box_mask, dv = generate_grid(b, d, box_edge_length)

print("\n--- 4. Calculating Slip Plane IDs ---")
# Assign a plane ID to every node for each of the four normal directions
slip_plane_ids = calculate_plane_ids(XX, YY, ZZ, b_vecs[:,0,0,:])

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
print("\n--- 5. Preparing Visualization ---")
# Choose which normal's planes to visualize (an integer from 0 to 3)
target_normal_idx = 3

# Call the visualization function to plot the results
visualize_planes(XX, YY, ZZ, box_mask, slip_plane_ids, target_normal_idx)


# =============================================================================
# POST-RUN ANALYSIS
#
# After running this script in an interactive console, you can inspect
# the variables directly. For example, try typing these commands into
# your console:
#
# >>> print(slip_plane_ids.shape)
# >>> print(f"Max plane ID for normal 0: {np.max(slip_plane_ids[:, :, :, 0])}")
#
# You can also re-run the visualization for a different normal:
# >>> visualize_planes(XX, YY, ZZ, box_mask, slip_plane_ids, target_normal_idx=1)
#
# =============================================================================
local_j_vec=np.array([np.cos(np.cos(np.pi/3)), np.sin(np.cos(np.pi/3)), 0])
local_i_vec=np.array([np.cos(np.cos(-np.pi/3)), np.sin(np.cos(-np.pi/3)), 0])
n_grid = int(n_grid)
local_XX= np.zeros((int(n_grid),int(n_grid)))
local_YY= np.zeros((int(n_grid),int(n_grid)))
for  i in range(n_grid):
    for j in range(n_grid):
        point= local_i_vec*i + local_j_vec*j
        local_XX[i,j] = point[0]
        local_YY[i,j] = point[1]

# computing the number of total slip planes existing in the box
num_slip_planes= np.max(slip_plane_ids[:,:,:,0])+np.max(slip_plane_ids[:,:,:,1])+np.max(slip_plane_ids[:,:,:,2])+np.max(slip_plane_ids[:,:,:,3])

# Making a dictionary putting the nodes with the same stack ID on the same nd array

# =============================================================================
# 6. GROUPING NODES BY SLIP SYSTEM
# =============================================================================
# Making a dictionary putting the nodes with the same stack ID on the same nd array

# Assuming XX, YY, ZZ, and slip_plane_ids are already defined in your previous code
# node_on_slip_sys needs to be initialized
node_on_slip_sys = {}

# --- 1. Populate the dictionary (Your loop) ---
for i in range(slip_plane_ids.shape[0]):
    for j in range(slip_plane_ids.shape[1]):
        for k in range(slip_plane_ids.shape[2]):
            # Use setdefault to initialize a list if the key doesn't exist yet
            node_on_slip_sys.setdefault((0, slip_plane_ids[i, j, k, 0]), []).append([i, j, k])
            node_on_slip_sys.setdefault((1, slip_plane_ids[i, j, k, 1]), []).append([i, j, k])
            node_on_slip_sys.setdefault((2, slip_plane_ids[i, j, k, 2]), []).append([i, j, k])
            node_on_slip_sys.setdefault((3, slip_plane_ids[i, j, k, 3]), []).append([i, j, k])


rss, active_s =calculate_rss_and_activity(load_direction, app_load, CRSS, b_vecs)

nodes_active= reconstruct_active_slip_planes(active_s, node_on_slip_sys, XX, YY, ZZ, (box_edge_length,box_edge_length,box_edge_length))
##############################
# now we will make the dislocation loops

# 1. Generate Loops (Pass nodes_active to filter valid stacks)
loops = generate_dislocation_loops_per_system(
    active_s,
    node_on_slip_sys,
    nodes_active,  # <--- NEW ARGUMENT
    XX, YY, ZZ,
    b_vecs,
    target_density,
    600*b,
    6000*b,
    b,
    box_edge_length
)

# 2. Compute Density Maps

# 2. Compute 3D Density/Theta Maps


# 2. Compute 3D Density/Theta Maps
QQ = compute_density_and_theta_maps(
    loops,
    nodes_active,
    b,
    nthetaintervals=nthetaintervals,
    std_dev_mult=600,
    cutoff_mult=5
)

# =============================================================================
# 3. SMEARING IN CONFIGURATIONAL SPACE (NEW STEP)
# =============================================================================

# Define standard deviation: 5 degrees in radians
std_dev_theta = 5 * np.pi / 180

# Call the new function
QQ_coarse = smear_configurational_density(QQ, nthetaintervals, std_dev_rad=std_dev_theta)
QQ_coarse = normalize_density_to_target(QQ_coarse, nodes_active, target_density)
# =============================================================================
# VISUALIZATION CHECK
# =============================================================================
if QQ_coarse:
    first_key = list(QQ_coarse.keys())[0]

    # Compare raw vs smeared for a specific pixel that has density
    raw_data = QQ[first_key]['density_map']
    smeared_data = QQ_coarse[first_key]['density_map']

    # Sum over spatial dimensions to see the angular distribution profile
    angular_profile_raw = np.sum(raw_data, axis=(0, 1))
    angular_profile_smeared = np.sum(smeared_data, axis=(0, 1))

    plt.figure(figsize=(10, 5))
    plt.plot(angular_profile_raw, label='Raw (Sharp)', linestyle='--')
    plt.plot(angular_profile_smeared, label='Smeared (Gaussian)', linewidth=2)
    plt.title(f"Angular Distribution Profile (Loop {first_key})")
    plt.xlabel("Theta Bin Index")
    plt.ylabel("Total Density")
    plt.legend()
    plt.show()

##########################################################
# stress computation section

C_tensor = get_elastic_constants(mu, nu)
eps_tensor, delta_tensor = get_epsilon_delta()
# Compute stress
interaction_stress_data = compute_interaction_stress_mura(
    QQ_coarse,  # Source densities
    C_tensor, eps_tensor, delta_tensor,
    nodes_active,  # Target meshes
    b_vecs,  # Slip system definitions
    b,  # Burgers magnitude
    mu,  # Mu
    nu,  # Nu
    a,  # 'a' parameter
    cut_off_dist=cutoff_dist_val,
    min_cut_off_density=min_density_val
)

# =============================================================================
# VISUALIZATION CHECK
# =============================================================================
if interaction_stress_data:
    # Example: Visualize stress for the first available system
    first_key = list(interaction_stress_data.keys())[3]
    data_entry = interaction_stress_data[first_key]

    stress_map = data_entry['stress_field']
    # Calculate magnitude for plotting
    stress_mag = stress_map

    # Mask 0 values (where there is no mesh) for better visualization
    stress_mag_masked = np.ma.masked_where(stress_mag == 0, stress_mag)

    plt.figure(figsize=(10, 8))
    plt.imshow(stress_mag_masked, origin='lower', cmap='plasma')
    plt.title(
        f"Interaction Stress Magnitude\nPlane {data_entry['slip_plane_normal_id']}, Stack {data_entry['slip_plane_stack_id']}, Sys {data_entry['slip_system_id']}")
    plt.colorbar(label='Stress Interaction (Pa)')
    plt.show()

    print(f"Computed stress for {len(interaction_stress_data)} target systems.")