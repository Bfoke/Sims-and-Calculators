import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar

'''

Chassis Coordinate System:
Origin at center of rear axle on ground plane
X - Forward
Y - Left
Z - Up

all local positions are in chassis frame coordinates
all unlabeled positions are in world frame coordinates

'''

class Chassis:
    def __init__(self, hardpoints):
        self.hardpoints = hardpoints  # dict of local-frame points
        self.R = np.eye(3)
        self.t = np.zeros(3)

    def to_world(self, p_local):
        return self.R @ p_local + self.t

    def set_pose(self, R, t):
        self.R = R
        self.t = t

class Wishbone:
    def __init__(self, front_local, rear_local, balljoint_local):
        #TODO: add rotation limits to restrict travel

        # local chassis pickup points in chassis frame coords
        self.front_local = front_local
        self.rear_local = rear_local
        self.balljoint_local = balljoint_local 
        
        # Zero pos. not mathematically necessary
        self.balljoint_0_local = balljoint_local

    def axis_of_rot(self): #helper function for rotation
        vec = self.front_local - self.rear_local
        mag = np.linalg.norm(vec)

        if mag == 0:
            raise ValueError("axis_of_rot unit vec length = 0")
        
        unit_vec = vec / mag
        return unit_vec
    
    def balljoint_pos(self, theta): 
        # move by angle theta from initialized position
        # return position in chassis frame
        axis = self.axis_of_rot()

        translated_point = self.balljoint_0_local - self.rear_local
        
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
        rotated_point = rotated_translated_point + self.rear_local

        return rotated_point #upright balljoint location
    
class Rack:
    def __init__(self, right_local, left_local, range, rotations, tie_rod_length):
        #lets you make angled and/or off center rack but don't do that
        #make a Rack object for the rear and don't use set_steer to mkae static toe links

        self.right_0 = right_local #0 positions, 3d point, chassis frame, do not modify
        self.left_0 = left_local

        self.range = range #distance a tie rod end moves when going from lock to lock, is equal to 2x the distance from centered steering wheel
        self.tie_rod_length = tie_rod_length #meters
        self.rotations = rotations #steering wheel rotations from lock to lock
    
    def set_steer(self, steer_pos):

        if steer_pos > (self.range/2):
            raise ValueError("steering exceeds rack range")
        
        # move tie rod ends in local chassis frame
        str = np.array([0,steer_pos,0])
        left = self.left_0 + str
        right = self.right_0 + str

        #return the positions
        return left, right
    
    # TODO: add another function later to return steering wheel theta as function of rack travel using range and rotations
    # input: current right or left tie rod pos
    # output: wheel theta

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
   
class Wheel:
    def __init__(self, center, axle_dir, radius):
        self.center = center          # 3D point
        self.axle_dir = axle_dir / np.linalg.norm(axle_dir) # 3D vector
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
    def __init__(self, upper_wb: Wishbone, lower_wb: Wishbone, upright: Upright, wheel: Wheel, rack: Rack, side):

        self.u_wb = upper_wb #wishbone object
        self.l_wb = lower_wb #wishbone object
        self.upright = upright #upright object
        self.rack = rack #object, steering rack rod ends if front or toe link locations if rear
        self.side = side # 0  for left, 1 for right
        self.wheel = wheel # wheel object

    def report_kingpin_inc(self, upper_bj, lower_bj): 
        #not in world frame, need to update or redo, also make it part of upright so it can call instance variables and dont need to pass in

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
    
    def camber_from_axle(axle_dir):
        n = axle_dir / np.linalg.norm(axle_dir)

        # project into YZ plane (remove X component)
        n_yz = np.array([0.0, n[1], n[2]])

        # reference vertical
        z = np.array([0.0, 0.0, 1.0])

        # angle magnitude
        cos_camber = np.dot(n_yz, z) / (
            np.linalg.norm(n_yz) * np.linalg.norm(z)
        )
        camber = np.arccos(np.clip(cos_camber, -1.0, 1.0))

        # sign: negative camber = top of wheel inboard
        # left side: +Y is outboard
        sign = -np.sign(n[1])
        return sign * camber

    def steer_from_axle(axle_dir):
        n = axle_dir / np.linalg.norm(axle_dir)

        # project into XY plane (remove Z component)
        n_xy = np.array([n[0], n[1], 0.0])

        # reference lateral direction (left)
        y = np.array([0.0, 1.0, 0.0])

        # signed angle using atan2
        cross = np.cross(y, n_xy)
        dot = np.dot(y, n_xy)

        steer = np.arctan2(cross[2], dot)

        return steer

def rigid_transform(original_pos, new_pos): 

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

        return R, t

def axis_of_rot(P1, P2): #helper function for rotation
        vec = P1 - P2
        mag = np.linalg.norm(vec)

        if mag == 0:
            raise ValueError("axis_of_rot unit vec length = 0")
        
        unit_vec = vec / mag
        return unit_vec

def lower_wb_solve(u_wb: Wishbone, l_wb: Wishbone, upright: Upright, upper_theta):
    """
    Pure solver for lower wishbone.
    Returns lower_theta and lower_balljoint_pos.
    """
    upper_pos = u_wb.balljoint_pos(upper_theta)
    joint_dist = upright.joint_dist

    def f(theta_l):
        lower_pos = l_wb.balljoint_pos(theta_l)
        return np.linalg.norm(upper_pos - lower_pos) - joint_dist

    result = root_scalar(f, method='brentq', bracket=(-0.5, 0.5))

    if not result.converged:
        raise RuntimeError("Lower wishbone solver failed")
    lower_theta = result.root
    return lower_theta, l_wb.balljoint_pos(lower_theta)

def solve_toe_link(
    upper_bj,
    lower_bj,
    rack_pos,
    upper_toe_dist,
    lower_toe_dist,
    tie_rod_length,
    prefer_forward=True
):
    """
    Solve toe link position by trilateration.

    All inputs are 3D numpy arrays in the same frame.
    Returns the physically valid toe link position.
    """

    P1 = np.asarray(upper_bj)
    P2 = np.asarray(lower_bj)
    P3 = np.asarray(rack_pos)

    # Basis construction
    ex = P2 - P1
    d = np.linalg.norm(ex)
    if d == 0:
        raise ValueError("Upper and lower ball joints coincide")

    ex /= d

    i = np.dot(ex, P3 - P1)
    temp = P3 - P1 - i * ex
    temp_norm = np.linalg.norm(temp)
    if temp_norm == 0:
        raise ValueError("Rack lies on BJ axis")

    ey = temp / temp_norm
    ez = np.cross(ex, ey)

    j = np.dot(ey, P3 - P1)

    # Coordinates in this frame
    x = (upper_toe_dist**2 - lower_toe_dist**2 + d**2) / (2 * d)
    y = (upper_toe_dist**2 - tie_rod_length**2 + i**2 + j**2 - 2 * i * x) / (2 * j)

    z_sq = upper_toe_dist**2 - x**2 - y**2
    if z_sq < 0:
        raise ValueError("No real toe-link solution")

    z = np.sqrt(z_sq)

    sol1 = P1 + x * ex + y * ey + z * ez
    sol2 = P1 + x * ex + y * ey - z * ez

    # Physical solution selection
    if prefer_forward:
        return sol1 if sol1[0] > sol2[0] else sol2
    else:
        return sol1 if sol1[0] < sol2[0] else sol2

def solve_corner(
    upper_theta, #radians
    steer, #meters
    corner_geom: Corner,   # contains u_wb, l_wb, upright, rack
    chassis_R=np.eye(3),
    chassis_t=np.zeros(3)
):
    """
    Returns all balljoints, toe link, and wheel geometry in world frame.
    """
    u_wb = corner_geom.u_wb
    l_wb = corner_geom.l_wb
    upright = corner_geom.upright
    rack = corner_geom.rack

    # Upper and lower wishbone
    upper_bj = u_wb.balljoint_pos(upper_theta)
    _, lower_bj = lower_wb_solve(u_wb, l_wb, upright, upper_theta)

    # Rack steering
    if corner_geom.side == 0:
        rack_pos = rack.left_0 + np.array([0, steer, 0])
    else:
        rack_pos = rack.right_0 + np.array([0, steer, 0])

    # Toe link
    toe_link = solve_toe_link(upper_bj, lower_bj, upright, rack_pos)

    # Apply chassis pose
    upper_bj_w = chassis_R @ upper_bj + chassis_t
    lower_bj_w = chassis_R @ lower_bj + chassis_t
    toe_link_w = chassis_R @ toe_link + chassis_t

    # Axle (wheel) vector
    axle_root_0 = upright.axle_root
    axle_tip_0 = upright.axle_tip
    axle_root_w = chassis_R @ axle_root_0 + chassis_t
    axle_tip_w = chassis_R @ axle_tip_0 + chassis_t
    axle_dir_w = axle_tip_w - axle_root_w
    axle_dir_w /= np.linalg.norm(axle_dir_w)

    # Wheel lowest point
    wheel_center = axle_root_w
    wheel_radius = upright.wheel_rad
    g = np.array([0.0, 0.0, -1.0])
    n = axle_dir_w
    d = g - np.dot(g, n) * n
    mag = np.linalg.norm(d)
    contact_point = wheel_center + (wheel_radius * d / mag if mag > 1e-9 else np.array([0,0,-wheel_radius]))

    return {
        "upper_bj": upper_bj_w,
        "lower_bj": lower_bj_w,
        "toe_link": toe_link_w,
        "axle_root": axle_root_w,
        "axle_dir": axle_dir_w,
        "wheel_center": wheel_center,
        "contact_point": contact_point,
    }