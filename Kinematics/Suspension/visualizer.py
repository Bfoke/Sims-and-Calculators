import pyvista as pv
import numpy as np
import jax.numpy as jnp

class SuspensionVisualizer:
    def __init__(self, params):
        self.params = params
        self.plotter = pv.Plotter(lighting="light_kit")
        self.meshes = {}
        # Set a clear background and view
        self.plotter.set_background("white")
        self.plotter.add_axes() # Useful for orientation

    def add_ground_plane(self):
        # Useful for visualizing contact patch movement
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=4, j_size=4)
        self.meshes["ground"] = self.plotter.add_mesh(plane, color="gray", opacity=0.3)

    # In visualizer.py
    def add_chassis_points(self, params): # Add params argument
        c = self.plotter
        r = 0.02
        # Use the passed params instead of self.params
        c.add_mesh(pv.Sphere(radius=r, center=params["u_front"]), color="white")
        c.add_mesh(pv.Sphere(radius=r, center=params["u_rear"]), color="white")
        c.add_mesh(pv.Sphere(radius=r, center=params["l_front"]), color="white")
        c.add_mesh(pv.Sphere(radius=r, center=params["l_rear"]), color="white")
        c.add_mesh(pv.Sphere(radius=r, center=params["rack_origin"]), color="yellow")
        
        # Draw rails
        c.add_mesh(pv.Line(params["u_front"], params["u_rear"]), color="white")
        c.add_mesh(pv.Line(params["l_front"], params["l_rear"]), color="white")

    # --- Geometry Initializers ---
    def setup_a_arms(self, p1, p2, bj_pos, color, name):
        """Creates triangular 'bones' for the wishbones"""
        # Upper wishbone mounts: params['u_origin'] and params['u_axis']
        mount_f = np.asarray(p1)
        mount_r = np.asarray(p2)
        # Placeholder balljoint position
        bj = np.asarray(bj_pos)
        
        # A triangle is just a custom PolyData
        # Points (F, R, BJ)
        nodes = np.stack([mount_f, mount_r, bj])
        # Face definition: one triangle, nodes 0, 1, 2
        face = np.array([3, 0, 1, 2])
        mesh = pv.PolyData(nodes, face)
        self.meshes[name] = self.plotter.add_mesh(mesh, color=color, opacity=0.7)

    def setup_hardpoints(self, bj_state, name):
        """Creates explicit, prominent spheres at the dynamic joints"""
        sphere = pv.Sphere(radius=0.02, center=bj_state)
        # Use a metallic/bright color to make them "steering axis" points
        self.meshes[name] = self.plotter.add_mesh(sphere, color="magenta", pbr=True, metallic=0.9, label=name)
    
    def setup_upright(self, ubj, lbj, toe, color, name):
        """Creates a triangular 'plate' representing the knuckle/upright."""
        nodes = np.stack([np.asarray(ubj), np.asarray(lbj), np.asarray(toe)])
        # One triangle, points 0, 1, 2
        face = np.array([3, 0, 1, 2])
        mesh = pv.PolyData(nodes, face)
        self.meshes[name] = self.plotter.add_mesh(mesh, color=color, opacity=0.9, pbr=True, metallic=0.5)

    def setup_tie_rod(self, rack, toe, name):
        """Creates a cylinder spanning two points"""
        line = np.array(toe) - np.array(rack)
        length = np.linalg.norm(line)
        center = np.array(rack) + line / 2
        cylinder = pv.Cylinder(center=center, direction=line, radius=0.015, height=length)
        self.meshes[name] = self.plotter.add_mesh(cylinder, color="gray")

    def setup_wheel_disc(self, center, axle_dir, radius, width, name):
        """Creates a simple disc/cylinder representing the wheel"""
        # A disc is a cylinder whose height is much smaller than its radius
        cylinder = pv.Cylinder(center=center, direction=axle_dir, radius=radius, height=width)
        # Make the wheel look like rubber/material, not metallic
        self.meshes[name] = self.plotter.add_mesh(cylinder, color="dimgray", opacity=0.6)

    # --- Scene Updaters (For Animation) ---
    def update_wishbone(self, name, p1_mount, p2_mount, bj_current):
        mount_f = np.asarray(p1_mount)
        mount_r = np.asarray(p2_mount)
        bj = np.asarray(bj_current)
        new_nodes = np.stack([mount_f, mount_r, bj])
        self.meshes[name].mapper.dataset.points = new_nodes

    def update_hardpoint(self, name, pos):
        # We must re-create the geometry object to change its position effectively
        new_geom = pv.Sphere(radius=0.015, center=pos)
        self.meshes[name].mapper.dataset.copy_from(new_geom)

    def update_upright(self, name, ubj, lbj, toe):
        """Moves the upright triangle as the suspension moves."""
        new_nodes = np.stack([np.asarray(ubj), np.asarray(lbj), np.asarray(toe)])
        self.meshes[name].mapper.dataset.points = new_nodes

    def update_tie_rod(self, name, p1_rack, p2_toe):
        line = np.array(p2_toe) - np.array(p1_rack)
        length = np.linalg.norm(line)
        center = np.array(p1_rack) + line / 2
        new_geom = pv.Cylinder(center=center, direction=line, radius=0.015, height=length)
        self.meshes[name].mapper.dataset.copy_from(new_geom)

    def update_wheel(self, name, center, axle_dir, radius, width):
        # Wheel must re-orient every frame
        new_geom = pv.Cylinder(center=center, direction=axle_dir, radius=radius, height=width)
        self.meshes[name].mapper.dataset.copy_from(new_geom)