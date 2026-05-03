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
    def add_chassis_points(self, params, suffix=""): 
        c = self.plotter
        r = 0.02
        c.add_mesh(pv.Sphere(radius=r, center=params["u_front"]), color="white")
        c.add_mesh(pv.Sphere(radius=r, center=params["u_rear"]), color="white")
        c.add_mesh(pv.Sphere(radius=r, center=params["l_front"]), color="white")
        c.add_mesh(pv.Sphere(radius=r, center=params["l_rear"]), color="white")
        
        # Give the rack a unique name based on the corner (e.g., rack_mesh_fr)
        rack_name = f"rack_mesh_{suffix}"
        self.meshes[rack_name] = c.add_mesh(
            pv.Sphere(radius=r, center=params["rack_origin"]), 
            color="yellow"
        )
        
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
        cylinder = pv.Cylinder(center=center, direction=line, radius=0.0125, height=length)
        self.meshes[name] = self.plotter.add_mesh(cylinder, color="gray")

    def setup_wheel_disc(self, center, axle_dir, radius, width, name):
        """Creates a simple disc/cylinder representing the wheel"""
        # A disc is a cylinder whose height is much smaller than its radius
        cylinder = pv.Cylinder(center=center, direction=axle_dir, radius=radius, height=width)
        # Make the wheel look like rubber/material, not metallic
        self.meshes[name] = self.plotter.add_mesh(cylinder, color="dimgray", opacity=0.6)

    def setup_instant_center(self, ic_pos, cp_pos, name):
        """Creates a sphere for the IC and a dashed line for the swing arm."""
        # The IC Sphere
        sphere = pv.Sphere(radius=0.03, center=ic_pos)
        self.meshes[f"{name}_pt"] = self.plotter.add_mesh(sphere, color="cyan", label="Instant Center")
        
        # The Swing Arm Line (CP to IC)
        line = pv.Line(cp_pos, ic_pos)
        self.meshes[f"{name}_line"] = self.plotter.add_mesh(line, color="cyan", line_width=1, opacity=0.5)

    def setup_pitch_center(self, pic_pos, cp_pos, name):
        """Creates a sphere for the Side-View IC and a line for anti-geometry visualization."""
        # The Pitch IC Sphere
        sphere = pv.Sphere(radius=0.03, center=pic_pos)
        self.meshes[f"{name}_pt"] = self.plotter.add_mesh(sphere, color="orange", label="Pitch IC")
        
        # The Swing Arm Line (Contact Patch to Pitch IC)
        line = pv.Line(cp_pos, pic_pos)
        self.meshes[f"{name}_line"] = self.plotter.add_mesh(line, color="orange", line_width=2, opacity=0.5)
    
    def setup_isa_axis(self, q, s, name, length=1.0):
        """Visualizes the 3D Screw Axis as a line passing through q."""
        # Create a line centered at q extending along s
        p1 = q - (s * length / 2)
        p2 = q + (s * length / 2)
        
        line = pv.Line(p1, p2)
        # We use a tube to make it more visible than a 1-pixel line
        self.meshes[f"{name}_axis"] = self.plotter.add_mesh(
            line.tube(radius=0.005), 
            color="lime", 
            label="Screw Axis (ISA)"
        )

    def add_chassis_skeleton_old(self, world_params_all):
        """
        Connects inboard hardpoints to visualize the chassis frame.
        world_params_all: dict containing 'fr', 'fl', 'rr', 'rl' world-space params
        """
        points = []
        lines = []
        
        def add_line(pt1, pt2):
            start_idx = len(points)
            points.extend([pt1, pt2])
            lines.extend([2, start_idx, start_idx + 1])

        for side in ['fr', 'fl', 'rr', 'rl']:
            p = world_params_all[side]
            # Connect Upper Arm mounts
            add_line(p['u_front'], p['u_rear'])
            # Connect Lower Arm mounts
            add_line(p['l_front'], p['l_rear'])
            # Connect Upper to Lower (Verticals)
            add_line(p['u_front'], p['l_front'])
            add_line(p['u_rear'], p['l_rear'])
            # Connect to Rack
            add_line(p['u_front'], p['rack_origin'])

        # Cross-Chassis Connections (Front and Rear)
        # Front Right to Front Left
        add_line(world_params_all['fr']['u_front'], world_params_all['fl']['u_front'])
        add_line(world_params_all['fr']['l_front'], world_params_all['fl']['l_front'])
        # Rear Right to Rear Left
        add_line(world_params_all['rr']['u_rear'], world_params_all['rl']['u_rear'])
        add_line(world_params_all['rr']['l_rear'], world_params_all['rl']['l_rear'])

        chassis_mesh = pv.PolyData(np.array(points))
        chassis_mesh.lines = np.array(lines)
        
        self.plotter.add_mesh(chassis_mesh, color="pink", line_width=3, label="Chassis Skeleton")

    def add_chassis_skeleton(self, world_params_all):
        """
        Connects inboard hardpoints to visualize the chassis frame and stores reference.
        """
        points, lines = self._build_chassis_geometry(world_params_all)
        
        chassis_mesh = pv.PolyData(np.array(points))
        chassis_mesh.lines = np.array(lines)
        
        # Store in meshes dict to allow updates
        self.meshes["chassis_skeleton"] = self.plotter.add_mesh(
            chassis_mesh, 
            color="pink", 
            line_width=3, 
            label="Chassis Skeleton"
        )
    
    def add_chassis_skeleton(self, world_params_all):
        """
        Connects inboard hardpoints to visualize the chassis frame and stores reference.
        """
        points, lines = self._build_chassis_geometry(world_params_all)
        
        chassis_mesh = pv.PolyData(np.array(points))
        chassis_mesh.lines = np.array(lines)
        
        # Store in meshes dict to allow updates
        self.meshes["chassis_skeleton"] = self.plotter.add_mesh(
            chassis_mesh, 
            color="pink", 
            line_width=3, 
            label="Chassis Skeleton"
        )

    def update_chassis_skeleton(self, world_params_all):
        """
        Updates the chassis frame positions during animation (heave/roll/pitch).
        """
        if "chassis_skeleton" not in self.meshes:
            return

        # Generate the new point list using the same order as the builder
        points, _ = self._build_chassis_geometry(world_params_all)
        
        # Update the points of the existing PolyData
        self.meshes["chassis_skeleton"].mapper.dataset.points = np.array(points)

    def _build_chassis_geometry(self, world_params_all):
        """Internal helper to ensure point order is consistent between add and update."""
        points = []
        lines = []
        
        def add_line(pt1, pt2):
            start_idx = len(points)
            points.extend([pt1, pt2])
            lines.extend([2, start_idx, start_idx + 1])

        # 1. Corner-specific verticals and longitudinals
        for side in ['fr', 'fl', 'rr', 'rl']:
            p = world_params_all[side]
            add_line(p['u_front'], p['u_rear'])
            add_line(p['l_front'], p['l_rear'])
            add_line(p['u_front'], p['l_front'])
            add_line(p['u_rear'], p['l_rear'])
            add_line(p['u_front'], p['rack_origin'])

        # 2. Lateral Cross-members
        # Front Cross
        add_line(world_params_all['fr']['u_front'], world_params_all['fl']['u_front'])
        add_line(world_params_all['fr']['l_front'], world_params_all['fl']['l_front'])
        # Rear Cross
        add_line(world_params_all['rr']['u_rear'], world_params_all['rl']['u_rear'])
        add_line(world_params_all['rr']['l_rear'], world_params_all['rl']['l_rear'])
        # Mid-chassis (connect front to rear)
        add_line(world_params_all['fr']['u_rear'], world_params_all['rr']['u_front'])
        add_line(world_params_all['fl']['u_rear'], world_params_all['rl']['u_front'])

        return points, lines

    def update_rack_displacement(self, name, original_pos, steer_val):
        """Moves the yellow rack sphere based on steering input."""
        # Calculate new 3D position
        new_pos = np.array(original_pos) + np.array([0, steer_val, 0])
        
        # Re-create the sphere geometry at the new center
        new_sphere = pv.Sphere(radius=0.02, center=new_pos)
        
        # Update the existing mesh in the plotter
        if name in self.meshes:
            self.meshes[name].mapper.dataset.copy_from(new_sphere)

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

    def update_instant_center(self, name, ic_pos, cp_pos):
        """Updates the IC position and the swing arm line."""
        # Update Sphere
        new_sphere = pv.Sphere(radius=0.03, center=ic_pos)
        self.meshes[f"{name}_pt"].mapper.dataset.copy_from(new_sphere)
        
        # Update Line
        new_line = pv.Line(cp_pos, ic_pos)
        self.meshes[f"{name}_line"].mapper.dataset.copy_from(new_line)

    def update_pitch_center(self, name, pic_pos, cp_pos):
        """Updates the Pitch IC position and its associated swing arm line."""
        new_sphere = pv.Sphere(radius=0.03, center=pic_pos)
        self.meshes[f"{name}_pt"].mapper.dataset.copy_from(new_sphere)
        
        new_line = pv.Line(cp_pos, pic_pos)
        self.meshes[f"{name}_line"].mapper.dataset.copy_from(new_line)
    
    def update_isa_axis(self, name, q, s, length=1.0):
        """Moves the ISA axis during animation."""
        p1 = q - (s * length / 2)
        p2 = q + (s * length / 2)
        
        new_line = pv.Line(p1, p2).tube(radius=0.005)
        self.meshes[f"{name}_axis"].mapper.dataset.copy_from(new_line)