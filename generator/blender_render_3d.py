import bpy
import matplotlib.pyplot as plt
from hydrogen import cartesian_prob, cartesian_prob_real
from get_render_radius import get_render_radius
import random 
import tqdm
import numpy as np

def render_3d(n, l, m, path, mode):
    print('Rendering ' + mode + ' 3d model for (' + str(n) + ', ' + str(l) +
          ', ' + str(m) + ')')

    render_radius = get_render_radius(n, l) + 2

    # width, height, and depth in number of steps
    s = 120

    # step = size of pixel in a_0
    step = 2 * render_radius / s

    # generate list of x, y, and z coordinates
    axis_set = [(float(i) - s / 2) * step for i in range(s + 1)]

    # lists to dump data
    x_data, y_data, z_data, p_data = [], [], [], []

    print('Calculating probabilities')
    # make data
    for x in tqdm.tqdm(axis_set, desc="Processing x-axis"):
        for y in axis_set:
            for z in axis_set:
                # calc p
                if mode == 'real':
                    p = cartesian_prob_real(n, l, m, x, y, z)
                elif mode == 'complex':
                    p = cartesian_prob(n, l, m, x, y, z)

                # append data
                p_data.append(p)
                x_data.append(x)
                y_data.append(y)
                z_data.append(z)

    print('Creating Blender objects')
    # Normalize probabilities
    p_data = np.array(p_data)
    p_data /= p_data.sum()

    # Sample 1000 points based on the probabilities
    indices = np.random.choice(len(p_data), size=1000, p=p_data)

    for i in indices:
        bpy.ops.object.empty_add(location=(x_data[i], y_data[i], z_data[i]))

    # Save the Blender file
    bpy.ops.wm.save_as_mainfile(filepath=path)
# Example usage
render_3d(3, 1, 1, 'path_to_save.blend', 'real')