import bpy
import numpy as np
import scipy.special as sp

# Property group to store orbital settings
class OrbitalSettings(bpy.types.PropertyGroup):
    n: bpy.props.IntProperty(
        name="n",
        description="Principal quantum number",
        default=2,
        min=1,
        max=10,
        update=lambda self, context: setattr(self, 'l', min(self.l, self.n - 1))
    ) # type: ignore
    
    l: bpy.props.IntProperty(
        name="l",
        description="Azimuthal quantum number",
        default=1,
        min=0,
        max=10,
        update=lambda self, context: setattr(self, 'm', max(min(self.m, self.l), -self.l))
    ) # type: ignore
    
    m: bpy.props.IntProperty(
        name="m",
        description="Magnetic quantum number",
        default=0,
        min=-10,
        max=10
    ) # type: ignore
    
    num_points: bpy.props.IntProperty(
        name="Particles",
        description="Number of particles to generate",
        default=10000,
        min=100,
        max=1000000
    ) # type: ignore

# Orbital generation function (same as before)
def hydrogen_orbital(n, l, m, num_points):
    points = []
    batch_size = max(10000, num_points)
    
    while len(points) < num_points:
        r = np.random.exponential(scale=n**2, size=batch_size)
        theta = np.arccos(2 * np.random.rand(batch_size) - 1)
        phi = np.random.uniform(0, 2 * np.pi, batch_size)
        
        rho = 2 * r / n
        radial_part = np.exp(-rho/2) * rho**l * sp.genlaguerre(n-l-1, 2*l+1)(rho)
        sph_harm = sp.sph_harm(m, l, phi, theta)
        prob_density = (radial_part * np.abs(sph_harm))**2
        
        prob_density /= np.max(prob_density)
        mask = np.random.rand(batch_size) < prob_density
        
        x = r[mask] * np.sin(theta[mask]) * np.cos(phi[mask])
        y = r[mask] * np.sin(theta[mask]) * np.sin(phi[mask])
        z = r[mask] * np.cos(theta[mask])
        
        new_points = np.column_stack((x, y, z))
        points.extend(new_points)
        
        if len(points) >= num_points:
            points = points[:num_points]
            break
            
    return np.array(points)

# Operator with validation
class GenerateOrbital(bpy.types.Operator):
    bl_idname = "object.generate_orbital"
    bl_label = "Generate Hydrogen Orbital"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.orbital_settings
        
        # Final validation
        if settings.l >= settings.n:
            self.report({'ERROR'}, "l must be less than n!")
            return {'CANCELLED'}
        if abs(settings.m) > settings.l:
            self.report({'ERROR'}, "|m| must be <= l!")
            return {'CANCELLED'}

        points = hydrogen_orbital(settings.n, settings.l, settings.m, settings.num_points)
        
        # Create object
        mesh = bpy.data.meshes.new(f"Orbital_{settings.n}{settings.l}{settings.m}")
        obj = bpy.data.objects.new(mesh.name, mesh)
        mesh.from_pydata(points.tolist(), [], [])
        mesh.update()
        
        # Add material
        mat = bpy.data.materials.new(name="Orbital")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = (0.34, 0.71, 1, 1)
        emission.inputs['Strength'].default_value = 1.5
        
        output = nodes.new('ShaderNodeOutputMaterial')
        mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
        
        obj.data.materials.append(mat)
        context.collection.objects.link(obj)
        
        return {'FINISHED'}

# Panel with input fields
class OrbitalPanel(bpy.types.Panel):
    bl_label = "Hydrogen Orbital"
    bl_idname = "PT_OrbitalPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Orbitals'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.orbital_settings
        
        # Quantum numbers
        box = layout.box()
        box.label(text="Quantum Numbers")
        box.prop(settings, 'n')
        box.prop(settings, 'l')
        box.prop(settings, 'm')
        
        # Particle settings
        layout.separator()
        box = layout.box()
        box.label(text="Particle Settings")
        box.prop(settings, 'num_points')
        
        # Generate button
        layout.separator()
        layout.operator("object.generate_orbital", 
                       text="Generate Orbital",
                       icon='PARTICLES')

# Registration
def register():
    bpy.utils.register_class(OrbitalSettings)
    bpy.utils.register_class(GenerateOrbital)
    bpy.utils.register_class(OrbitalPanel)
    bpy.types.Scene.orbital_settings = bpy.props.PointerProperty(type=OrbitalSettings)

def unregister():
    bpy.utils.unregister_class(OrbitalSettings)
    bpy.utils.unregister_class(GenerateOrbital)
    bpy.utils.unregister_class(OrbitalPanel)
    del bpy.types.Scene.orbital_settings

if __name__ == "__main__":
    register()