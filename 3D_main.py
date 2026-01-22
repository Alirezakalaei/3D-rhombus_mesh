# main.py

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
    compute_dislocation_density_maps
)

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
load_direction= np.array([0, 0, 1])
b_vecs = calculate_tetrahedron_slip_systems(b)
box_edge_length = 20000 * b
n_grid = np.ceil(box_edge_length / d)
# =============================================================================
# 2. EXECUTE COMPUTATIONAL STEPS
# =============================================================================
print("\n--- 2. Generating Grid ---")
# Generate the grid points and the mask for the simulation box
XX, YY, ZZ, box_mask = generate_grid(b, d, box_edge_length)

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

nodes_active= reconstruct_active_slip_planes(active_s, node_on_slip_sys, XX, YY, ZZ)
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
from fcc_lib import compute_dislocation_density_maps

QQ = compute_dislocation_density_maps(
    loops,
    nodes_active,
    b,
    std_dev_mult=600,
    cutoff_mult=5
)

# Optional: Visualize one result
if QQ:
    import matplotlib.pyplot as plt
    first_loop = list(QQ.values())[0]
    plt.imshow(first_loop['density_map'], origin='lower')
    plt.title("Dislocation Density Map")
    plt.colorbar()
    plt.show()