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
    s = 512

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
    indices = np.random.choice(len(p_data), size=10000, p=p_data)

    # Clear all existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Create "Empty" collection
    empty_collection = bpy.data.collections.new("Empty")
    bpy.context.scene.collection.children.link(empty_collection)

    # Create "Source" collection
    source_collection = bpy.data.collections.new("Source")
    bpy.context.scene.collection.children.link(source_collection)

    # Create new empties and link them to the "Empty" collection
    for i in indices:
        empty = bpy.data.objects.new("Empty", None)
        # Add a small random displacement to each particle's position
        displacement = np.random.uniform(-step/2, step/2, 3)
        empty.location = (x_data[i] + displacement[0], y_data[i] + displacement[1], z_data[i] + displacement[2])
        empty.instance_collection = source_collection
        empty_collection.objects.link(empty)

    # Save the Blender file
    bpy.ops.wm.save_as_mainfile(filepath=path)

# Example usage
render_3d(3, 1, 1, 'path_to_save.blend', 'real')