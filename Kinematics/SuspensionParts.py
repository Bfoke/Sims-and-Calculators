import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar
'''
Coordinate system for everything is:
Origin at center of rear axle on ground plane
X - Forward
Y - Left
Z - Up

'''

class Wishbone:
    def __init__(self, front: np.array, rear: np.array, balljoint: np.array):
        #add rotation limits to restrict travel
        self.front = front # pickup points on chassis
        self.rear = rear
        self.balljoint = balljoint

        # zero-angle reference
        self.balljoint_0 = np.copy(balljoint)
        self.current_theta = 0     

    def axis_of_rot(self):
        vec = self.front - self.rear
        mag = np.linalg.norm(vec)

        if mag == 0:
            raise ValueError("axis_of_rot unit vec length = 0")
        
        unit_vec = vec / mag
        return unit_vec
    
    def balljoint_pos(self):
        return self.balljoint
    
    def rotation(self, theta): #move by angle theta
        
        axis = self.axis_of_rot()

        translated_point = self.balljoint - self.rear
        
        ux, uy, uz = axis
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        one_minus_cos = 1 - cos_t

        R = np.array([
            [cos_t + ux**2 * one_minus_cos,
            ux * uy * one_minus_cos - uz * sin_t,
            ux * uz * one_minus_cos + uy * sin_t],

            [uy * ux * one_minus_cos + uz * sin_t,
            cos_t + uy**2 * one_minus_cos,
            uy * uz * one_minus_cos - ux * sin_t],

            [uz * ux * one_minus_cos - uy * sin_t,
            uz * uy * one_minus_cos + ux * sin_t,
            cos_t + uz**2 * one_minus_cos]
        ])

        # rotate the translated point
        rotated_translated_point = R @ translated_point

        # translate back to the original coordinate system
        rotated_point = rotated_translated_point + self.rear
        
        # change instance variable of balljoint location
        self.balljoint = rotated_point 

        return rotated_point #upright balljoint location
    
    def set_angle(self, theta):
        """
        Sets the wishbone to an absolute angle theta (radians)
        relative to the initial zero-angle position.
        """
        # Determine the rotation needed from zero-angle
        delta_theta = theta - getattr(self, "current_theta", 0)

        # Temporarily reset balljoint to zero position
        self.balljoint = np.copy(self.balljoint_0)

        # Rotate by delta_theta using your existing rotation method
        self.rotation(delta_theta)

        # Update current angle
        self.current_theta = theta

        return self.balljoint

class Upright:
    def __init__(self, upper_balljoint, lower_balljoint, toe_link, axle_root, axle_tip, wheel_rad):
        # roots are points along the vector between upper and lower balljoint that the orthogonal projection of toe_link and axle tip go to
        # they are used to locate the toe_link and axle_tip relative to the balljoint vector

        #ball joints
        self.upper_balljoint = upper_balljoint
        self.lower_balljoint = lower_balljoint
        self.toe_link = toe_link

        #0 positions of joints
        self.upper_balljoint_0 = np.copy(upper_balljoint)
        self.lower_balljoint_0 = np.copy(lower_balljoint)
        self.toe_link_0 = np.copy(toe_link)
        self.upright_0 = self.upper_balljoint_0, self.lower_balljoint_0, self.toe_link_0

        #wheel geometry
        self.axle_root = axle_root # put this on centerline of wheel + tire
        self.axle_tip = axle_tip
        self.axle_vec = axle_tip - axle_root
        self.wheel_rad = wheel_rad #wheel is 2d disc orthogonal to axel_vec
        
        # distances for solving rotations
        self.joint_dist = np.linalg.norm(upper_balljoint - lower_balljoint)
        self.upper_toe_dist = np.linalg.norm(upper_balljoint - toe_link)
        self.lower_toe_dist = np.linalg.norm(lower_balljoint - toe_link)

    def wheel(self): #define wheel object from upright inputs
        axle_dir = self.axle_tip - self.axle_root
        axle_dir /= np.linalg.norm(axle_dir)
        center = self.axle_root
        return Wheel(center, axle_dir, self.wheel_rad)
    
    def upright_pos(self):
        return self.upper_balljoint, self.lower_balljoint, self.toe_link, self.axle_root, self.axle_tip

class Rack:
    def __init__(self, right, left, range, rotations, tie_rod_length):
        #lets you make angled and/or off center rack but don't do that
        #don't articulate the rack to make static toe link locations for the rear

        self.right = right #3d point
        self.left = left #3d point

        self.right_0 = right #0 positions
        self.left_0 = left

        self.range = range #distance a tie rod end moves when going from lock to lock, is equal to 2x the distance from centered steering wheel
        self.tie_rod_length = tie_rod_length #meters
        self.rotations = rotations #steering wheel rotations from lock to lock

    def steer(self, steer_dist):
        # positive steer_dist = left turn
        # negative steer_dist = right turn
        # matches with global coordinate system
        
        str = np.array([0,steer_dist,0])
        right_rod_end = self.right + str
        left_rod_end = self.left + str

        return left_rod_end, right_rod_end
    
    def set_steer(self, steer_pos):

        if steer_pos > (self.range/2):
            raise ValueError("steering exceeds rack range")
        
        str = np.array([0,steer_pos,0])
        right_rod_end = self.right + str
        left_rod_end = self.left + str

        #set instance variables
        self.right = right_rod_end
        self.left = left_rod_end

        #return the positions
        return left_rod_end, right_rod_end
    
    # TODO: add another function later to return steering wheel theta as function of rack travel using range and rotations
    # input: current right or left tie rod pos 
    # output: wheel theta
    # this might not be useful since will be picking ots rack

class Wheel:
    def __init__(self, center, axle_dir, radius):
        self.center = center          # 3D
        self.axle_dir = axle_dir / np.linalg.norm(axle_dir)
        self.radius = radius

    def lowest_point(self):
        g = np.array([0.0, 0.0, -1.0])
        n = self.axle_dir

        d = g - np.dot(g, n) * n
        mag = np.linalg.norm(d)

        if mag < 1e-9:
            # wheel plane vertical
            return self.center + np.array([0, 0, -self.radius])

        d_hat = d / mag
        return self.center + self.radius * d_hat


class Corner:
    def __init__(self, upper_wb, lower_wb, upright, wheel, rack, side):
        self.u_wb = upper_wb #wishbone object
        self.l_wb = lower_wb #wishbone object
        self.upright = upright #upright object
        self.rack = rack #object, steering rack rod ends if front or toe link locations if rear
        self.side = side # 0  for left, 1 for right
        self.wheel = wheel

    def wishbone_travel(self, upper_theta):# , theta_bounds=(-0.5, 0.5)):
        """
        Solve for lower wishbone theta such that the distance between upper and
        lower balljoints equals the upright joint distance.
        """

        # Fixed position of upper balljoint
        upper_pos = self.u_wb.set_angle(upper_theta)
        joint_dist = self.upright.joint_dist

        def f(theta_l):
            lower_pos = self.l_wb.set_angle(theta_l)
            dist = jnp.linalg.norm(upper_pos - lower_pos)
            return dist - joint_dist

        # Use root-finding to solve f(theta) = 0
        result = root_scalar(f, method='brentq')# , bracket=theta_bounds)

        if not result.converged:
            raise RuntimeError("Failed to solve for lower wishbone theta")

        best_theta = result.root

        return self.l_wb.set_angle(best_theta)
    
    def toe_link_pos_solve(self):
        # use "trilaterate" method to find position of toe link on the upright

        #get upper wishbone balljoint pos
        upper_pos = self.u_wb.balljoint
        upper_dist = self.upright.upper_toe_dist
        
        #get lower wishbone balljoint pos
        lower_pos = self.l_wb.balljoint
        lower_dist = self.upright.lower_toe_dist

        #rack_pos gives third point
        tie_rod_dist = self.rack.tie_rod_length

        if self.side == 0: # left or right side
            rack_pos = self.rack.left
        else:
            rack_pos = self.rack.right

        P1, P2, P3 = map(np.array, (upper_pos, lower_pos, rack_pos))

        # Create unit vectors
        ex = (P2 - P1)
        ex /= np.linalg.norm(ex)
        i = np.dot(ex, P3 - P1)
        temp = P3 - P1 - i * ex
        ey = temp / np.linalg.norm(temp)
        ez = np.cross(ex, ey)

        d = np.linalg.norm(P2 - P1)
        j = np.dot(ey, P3 - P1)

        # Coordinates in the new system
        x = (upper_dist**2 - lower_dist**2 + d**2) / (2 * d)
        y = (upper_dist**2 - tie_rod_dist**2 + i**2 + j**2 - 2 * i * x) / (2 * j)

        # Solve for z^2, and check if real solution exists
        z_sq = upper_dist**2 - x**2 - y**2
        if z_sq < 0:
            raise ValueError("No real solution exists (spheres don't intersect)")

        z = np.sqrt(z_sq)

        # Convert back to original coordinate system
        result1 = P1 + x * ex + y * ey + z * ez
        result2 = P1 + x * ex + y * ey - z * ez

        # this returns the solution for the tie rod link location where the tie rod is more forward 
        # corresponding with rack in front of steering axis

        # change this inequality if setting the rack behind the steering axis
        return result1 if result1[0] > result2[0] else result2 
    
    def rigid_transform(original_pos, new_pos): 
        # original and new pos are upper wishbone, lower wishbone, and toe link positions before and after move, (9 total values each)
        # for moving wheel/tire after suspension is articulated

        """
        Computes the rigid transformation (rotation + translation) that aligns P to Q.

        Parameters:
            P: (3, 3) numpy array of source points
            Q: (3, 3) numpy array of destination points

        Returns:
            R: (3, 3) rotation matrix
            t: (3,) translation vector
        """
        # Centroids
        P_centroid = np.mean(original_pos, axis=0)
        Q_centroid = np.mean(new_pos, axis=0)

        # Center the point sets
        P_centered = original_pos - P_centroid
        Q_centered = new_pos - Q_centroid

        # Kabsch algorithm to find rotation
        H = np.dot(P_centered.T, Q_centered)
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
        D = np.diag([1, 1, d])
        R = np.dot(Vt.T, np.dot(D, U.T))

        # Translation
        t = Q_centroid - np.dot(P_centroid, R)

        return R, t #apply this rotation and translation to full upright assembly including tire

    def articulate(self, upper_wb_theta, steer_dist):
        """
        Articulates the corner given:
        - upper wishbone rotation
        - rack travel

        Updates upright + wheel pose and returns wheel ground contact point.
        """

        # wishbone positions
        upper_bj = self.u_wb.set_angle(upper_wb_theta)
        lower_bj = self.wishbone_travel(upper_wb_theta)

        # steering
        self.rack.set_steer(steer_dist)

        # solve toe link position
        toe_link = self.toe_link_pos_solve()

        # rigid transform for upright/ wheel
        upper_bj_0, lower_bj_0, toe_link_0 = self.upright.upright_0

        P0 = np.vstack([upper_bj_0, lower_bj_0, toe_link_0])
        P1 = np.vstack([upper_bj,   lower_bj,   toe_link])

        R, t = self.rigid_transform(P0, P1)

        # change upright instance vars
        self.upright.upper_balljoint = upper_bj
        self.upright.lower_balljoint = lower_bj
        self.upright.toe_link = toe_link

        self.upright.axle_root = R @ self.upright.axle_root + t
        self.upright.axle_tip  = R @ self.upright.axle_tip  + t

        # --- 4. Analytic wheel model ---
        axle_vec = self.upright.axle_tip - self.upright.axle_root
        axle_dir = axle_vec / np.linalg.norm(axle_vec)
        wheel_center = self.upright.axle_root
        wheel_radius = self.upright.wheel_rad

        contact_point = self.wheel.wheel_lowest_point(
            wheel_center,
            axle_dir,
            wheel_radius
        )

        return {
            "upper_bj": upper_bj,
            "lower_bj": lower_bj,
            "toe_link": toe_link,
            "wheel_center": wheel_center,
            "axle_dir": axle_dir,
            "contact_point": contact_point
        }


    def report_kingpin_inc(self, upper_bj, lower_bj):
        # kingpin axis vector
        bj_vec = upper_bj - lower_bj

        # reference vertical vector (pointing up from lower BJ)
        vertical_vec = np.array([0, 0, 1])

        # define plane normal: mid-plane = xz-plane → normal = y-axis
        mid_plane_normal = np.array([0, 1, 0], dtype=float)
        mid_plane_normal /= np.linalg.norm(mid_plane_normal)  # ensure unit normal

        # projection onto plane: v_proj = v - (v·n)n
        bj_proj = bj_vec - np.dot(bj_vec, mid_plane_normal) * mid_plane_normal
        vert_proj = vertical_vec - np.dot(vertical_vec, mid_plane_normal) * mid_plane_normal

        # angle between projected vectors
        cos_theta = np.dot(bj_proj, vert_proj) / (
            np.linalg.norm(bj_proj) * np.linalg.norm(vert_proj)
        )

        # numerical safety
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        return np.arccos(cos_theta)
    
    def report_caster(self, upper_bj, lower_bj):
        # kingpin axis vector
        bj_vec = upper_bj - lower_bj

        # reference vertical vector
        vertical_vec = np.array([0.0, 0.0, 1.0])

        # plane normal for camber: yz-plane → normal = x-axis
        plane_normal = np.array([1.0, 0.0, 0.0])
        plane_normal /= np.linalg.norm(plane_normal)   # ensure unit normal

        # project kingpin axis and vertical vector onto the plane
        bj_proj = bj_vec - np.dot(bj_vec, plane_normal) * plane_normal
        vert_proj = vertical_vec - np.dot(vertical_vec, plane_normal) * plane_normal

        # compute camber angle between the two projected vectors
        cos_theta = np.dot(bj_proj, vert_proj) / (
            np.linalg.norm(bj_proj) * np.linalg.norm(vert_proj)
        )

        # numerical stability
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        return np.arccos(cos_theta)