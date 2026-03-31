import pyvista as pv
import numpy as np
import jax.numpy as jnp

class SuspensionVisualizer:
    def __init__(self, params):
        self.params = params
        self.plotter = pv.Plotter(lighting="light_kit")
        self.meshes = {}
        # Set a clear background and view
        self.plotter.set_background("black")
        self.plotter.add_axes() # Useful for orientation

    def add_ground_plane(self):
        # Useful for visualizing contact patch movement
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=500, j_size=1000)
        self.meshes["ground"] = self.plotter.add_mesh(plane, color="gray", opacity=0.3)

    def add_chassis_points(self):
        # Fixed points on the chassis
        c = self.plotter # Shorthand
        c.add_mesh(pv.Sphere(radius=8, center=self.params["u_origin"]), color="white", label="U Mount F")
        c.add_mesh(pv.Sphere(radius=8, center=self.params["u_axis"]), color="white", label="U Mount R")
        c.add_mesh(pv.Sphere(radius=8, center=self.params["l_origin"]), color="white", label="L Mount F")
        c.add_mesh(pv.Sphere(radius=8, center=self.params["l_axis"]), color="white", label="L Mount R")
        c.add_mesh(pv.Sphere(radius=8, center=self.params["rack_origin"]), color="yellow", label="Rack Tie-In")

    # --- Geometry Initializers ---
    def setup_a_arms(self, p1, p2, color, name):
        """Creates triangular 'bones' for the wishbones"""
        # Upper wishbone mounts: params['u_origin'] and params['u_axis']
        mount_f = np.asarray(p1)
        mount_r = np.asarray(p2)
        # Placeholder balljoint position
        bj = np.asarray([mount_f[0], mount_f[1], mount_f[2] + 50]) 
        
        # A triangle is just a custom PolyData
        # Points (F, R, BJ)
        nodes = np.stack([mount_f, mount_r, bj])
        # Face definition: one triangle, nodes 0, 1, 2
        face = np.array([3, 0, 1, 2])
        mesh = pv.PolyData(nodes, face)
        self.meshes[name] = self.plotter.add_mesh(mesh, color=color, opacity=0.7)

    def setup_hardpoints(self, bj_state, name):
        """Creates explicit, prominent spheres at the dynamic joints"""
        sphere = pv.Sphere(radius=15, center=bj_state)
        # Use a metallic/bright color to make them "steering axis" points
        self.meshes[name] = self.plotter.add_mesh(sphere, color="magenta", pbr=True, metallic=0.9, label=name)

    def setup_tie_rod(self, rack, toe, name):
        """Creates a cylinder spanning two points"""
        line = np.array(toe) - np.array(rack)
        length = np.linalg.norm(line)
        center = np.array(rack) + line / 2
        cylinder = pv.Cylinder(center=center, direction=line, radius=6, height=length)
        self.meshes[name] = self.plotter.add_mesh(cylinder, color="gray")

    def setup_wheel_disc(self, center, axle_dir, radius, width, name):
        """Creates a simple disc/cylinder representing the wheel"""
        # A disc is a cylinder whose height is much smaller than its radius
        cylinder = pv.Cylinder(center=center, direction=axle_dir, radius=radius, height=width)
        # Make the wheel look like rubber/material, not metallic
        self.meshes[name] = self.plotter.add_mesh(cylinder, color="dimgray", opacity=0.8)

    # --- Scene Updaters (For Animation) ---
    def update_wishbone(self, name, p1_mount, p2_mount, bj_current):
        mount_f = np.asarray(p1_mount)
        mount_r = np.asarray(p2_mount)
        bj = np.asarray(bj_current)
        new_nodes = np.stack([mount_f, mount_r, bj])
        self.meshes[name].resource.points = new_nodes

    def update_hardpoint(self, name, pos):
        # We must re-create the geometry object to change its position effectively
        new_geom = pv.Sphere(radius=15, center=pos)
        self.meshes[name].resource.copy_from(new_geom)

    def update_tie_rod(self, name, p1_rack, p2_toe):
        line = np.array(p2_toe) - np.array(p1_rack)
        length = np.linalg.norm(line)
        center = np.array(p1_rack) + line / 2
        new_geom = pv.Cylinder(center=center, direction=line, radius=6, height=length)
        self.meshes[name].resource.copy_from(new_geom)

    def update_wheel(self, name, center, axle_dir, radius, width):
        # Wheel must re-orient every frame
        new_geom = pv.Cylinder(center=center, direction=axle_dir, radius=radius, height=width)
        self.meshes[name].resource.copy_from(new_geom)