import os
import random
import numpy as np
import tqdm
import concurrent.futures
import bpy

# Import heavy computation functions and modules that are safe for spawning.
from hydrogen import cartesian_prob, cartesian_prob_real
from get_render_radius import get_render_radius
# Note: We do NOT import bpy here.

def compute_probs_for_x(x, axis_set, n, l, m, mode):
    """
    Computes probability values for all y and z for a fixed x coordinate.
    Returns lists for x, y, z coordinates and the computed probability values.
    """
    local_x, local_y, local_z, local_p = [], [], [], []
    for y in axis_set:
        for z in axis_set:
            if mode == 'real':
                p = cartesian_prob_real(n, l, m, x, y, z)
            elif mode == 'complex':
                p = cartesian_prob(n, l, m, x, y, z)
            else:
                raise ValueError("Invalid mode provided: choose 'real' or 'complex'")
            local_x.append(x)
            local_y.append(y)
            local_z.append(z)
            local_p.append(p)
    return local_x, local_y, local_z, local_p

def create_blender_objects(x_data, y_data, z_data, p_data):
    # Normalize probabilities
    p_data = np.array(p_data, dtype=np.float64)
    p_data /= p_data.sum()

    # Sample 5000 points based on the computed probabilities
    indices = np.random.choice(len(p_data), size=5000, p=p_data)

    for i in tqdm.tqdm(indices, desc="Adding Blender objects"):
        bpy.ops.object.empty_add(location=(x_data[i], y_data[i], z_data[i]))

    # Ask for the file name
    file_name = input("Enter the file name to save (press Enter to use default): ").strip()
    if not file_name:
        file_name = f"notusre.blend"

    base_name, extension = os.path.splitext(file_name)
    counter = 1
    while os.path.exists(file_name):
        file_name = f"{base_name}_{counter}{extension}"
        counter += 1

    # Save the Blender file
    bpy.ops.wm.save_as_mainfile(filepath=file_name)

def render_3d(n, l, m, mode):
    # Import bpy inside the function so that child processes (spawned by multiprocessing)
    # never attempt to import it.
    import bpy

    print(f"Rendering {mode} 3d model for ({n}, {l}, {m})")

    render_radius = get_render_radius(n, l) + 2

    # Number of steps along each axis
    s = 256
    step = 2 * render_radius / s
    axis_set = [(float(i) - s / 2) * step for i in range(s + 1)]

    # Prepare lists to gather data
    x_data, y_data, z_data, p_data = [], [], [], []

    print('Calculating probabilities in parallel')

    # Use ProcessPoolExecutor to parallelize over the x-axis.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(compute_probs_for_x, x, axis_set, n, l, m, mode)
            for x in axis_set
        ]

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures),
                                total=len(futures),
                                desc="Processing x-axis"):
            try:
                local_x, local_y, local_z, local_p = future.result()
                x_data.extend(local_x)
                y_data.extend(local_y)
                z_data.extend(local_z)
                p_data.extend(local_p)
            except Exception as e:
                print("A subprocess raised an exception:", e)

    print('Creating Blender objects')
    create_blender_objects(x_data, y_data, z_data, p_data)

def main():
    # Only call Blender-dependent code here.
    render_3d(3, 1, 1, 'real')

if __name__ == '__main__':
    main()
