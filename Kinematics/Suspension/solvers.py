import numpy as np
from .geometry import Wishbone, Upright, Wheel, Rack, Corner, Chassis
import jax.numpy as jnp
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

################################################
# Functions for manipulating suspension geometry
################################################

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

def lower_wb_solve(u_wb: Wishbone, l_wb: Wishbone, upright: Upright, upper_theta):
    """
    Pure solver for lower wishbone.
    Returns lower_theta and lower_balljoint_pos.
    """
    upper_pos = u_wb.set_balljoint_pos(upper_theta)
    joint_dist = upright.joint_dist

    def f(theta_l):
        lower_pos = l_wb.set_balljoint_pos(theta_l)
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
    forward_rack=True
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
    if forward_rack:
        return sol1 if sol1[0] > sol2[0] else sol2
    else:
        return sol1 if sol1[0] < sol2[0] else sol2
    
def solve_corner(
    upper_theta, #radians
    steer, #meters
    corner_geom: Corner,   # contains u_wb, l_wb, upright, rack
):
    """
    Returns all balljoints, toe link, and wheel geometry in chassis frame.
    """
    u_wb = corner_geom.u_wb
    l_wb = corner_geom.l_wb
    upright = corner_geom.upright
    rack = corner_geom.rack

    # set upper and lower wishbones
    upper_bj = u_wb.set_balljoint_pos(upper_theta)
    _, lower_bj = lower_wb_solve(u_wb, l_wb, upright, upper_theta)

    # set steering rack (do nothing for rear)
    if corner_geom.side == 0:
        rack_pos = rack.left_0 + np.array([0, steer, 0])
    else:
        rack_pos = rack.right_0 + np.array([0, steer, 0])

    # solve for toe link given wishbone balljoints and rack position
    toe_link = solve_toe_link(upper_bj, lower_bj, rack_pos, upright.upper_toe_dist, upright.lower_toe_dist, rack.tie_rod_length)

    #transform upright/ wheel positions to fit new position
    # Upright reference points (local / zero pose)
    P0 = np.vstack([
        upright.upper_balljoint_0,
        upright.lower_balljoint_0,
        upright.toe_link_0,
    ])

    # Upright solved points (current pose)
    P1 = np.vstack([
        upper_bj,
        lower_bj,
        toe_link,
    ])

    R_upright, t_upright = rigid_transform(P0, P1)

    axle_root = R_upright @ upright.axle_root + t_upright
    axle_tip  = R_upright @ upright.axle_tip  + t_upright
    axle_dir  = axle_tip - axle_root
    axle_dir /= np.linalg.norm(axle_dir)

    # Wheel lowest point / contact point
    wheel_center = axle_root
    wheel_radius = upright.wheel_rad 
    g = np.array([0.0, 0.0, -1.0]) 
    n = axle_dir
    d = g - np.dot(g, n) * n 
    mag = np.linalg.norm(d) 
    contact_point = wheel_center + (wheel_radius * d / mag if mag > 1e-9 else np.array([0,0,-wheel_radius]))

    return {
        "upper_bj": upper_bj,
        "lower_bj": lower_bj,
        "toe_link": toe_link,
        "axle_root": axle_root,
        "axle_dir": axle_dir,
        "wheel_center": wheel_center,
        "contact_point": contact_point
    }

##############################################
# Functions for evaluating suspension geometry
##############################################

def project_YZ(p):
    """Project 3D point into front view (YZ plane)."""
    return np.array([p[1], p[2]])  # (Y, Z)

def line_intersection_2d(p1, p2, p3, p4):
    """
    Intersection of two 2D lines:
    line 1: p1 -> p2
    line 2: p3 -> p4
    Returns None if parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if abs(denom) < 1e-9:
        return None

    px = ((x1*y2 - y1*x2)*(x3 - x4) -
          (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) -
          (y1 - y2)*(x3*y4 - y3*x4)) / denom

    return np.array([px, py])

def front_view_instant_center(u_wb: Wishbone, l_wb: Wishbone,
                              upper_bj, lower_bj):
    """
    Returns IC in chassis frame (3D).
    """

    # Upper arm line (projected)
    u_in = project_YZ(u_wb.front_local)
    u_out = project_YZ(upper_bj)

    # Lower arm line (projected)
    l_in = project_YZ(l_wb.front_local)
    l_out = project_YZ(lower_bj)

    ic_yz = line_intersection_2d(u_in, u_out, l_in, l_out)
    if ic_yz is None:
        return None

    # Back to 3D (X arbitrary → use 0)
    return np.array([0.0, ic_yz[0], ic_yz[1]])

def roll_center_from_two_corners(
    ic_left, contact_left,
    ic_right, contact_right
):
    """
    True geometric roll center:
    intersection of force lines from contact patches to instant centers
    """

    if ic_left is None or ic_right is None:
        return None

    # Project to front view
    Lc = project_YZ(contact_left)
    Li = project_YZ(ic_left)

    Rc = project_YZ(contact_right)
    Ri = project_YZ(ic_right)

    rc_yz = line_intersection_2d(Lc, Li, Rc, Ri)
    if rc_yz is None:
        return None

    return np.array([0.0, rc_yz[0], rc_yz[1]])

def report_kingpin_inc(upper_bj, lower_bj): 
        
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

def report_caster(upper_bj, lower_bj):
        # only valid if vehicle pose is on ground and facing positive X direction
        # TODO: add chassis R and T to make it always valid
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

def report_camber_from_axle(axle_dir):

    # report camber in chassis frame or world frame depending on which axle_dir is passed
    # TODO: add ability to pass in world frame R and t matrices so can get camber in world frame
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

def report_toe_from_axle(axle_dir):

    # report toe in chassis frame, only works if car is facing positive X direction
    # TODO: add chassis R and T to compensate for chassis pose and report toe relative to chassis body not the axis

    n = axle_dir / np.linalg.norm(axle_dir)

    # project into XY plane (remove Z component)
    n_xy = np.array([n[0], n[1], 0.0])

    # reference lateral direction (left)
    y = np.array([0.0, 1.0, 0.0])

    # signed angle using atan2
    cross = np.cross(y, n_xy)
    dot = np.dot(y, n_xy)

    toe = np.arctan2(cross[2], dot)

    return toe

def solve_and_measure_corner(theta, steer, corner):
    #funciton for performing sweeps and measuring changes on one corner
    # TODO: scrub radius, mechanical trail
    res = solve_corner(theta, steer, corner)

    axle_dir = res["axle_dir"]
    wheel_center = res["wheel_center"]
    contact = res["contact_point"]
    upper_bj = res["upper_bj"]
    lower_bj = res["lower_bj"]

    camber = report_camber_from_axle(axle_dir)
    toe = report_toe_from_axle(axle_dir)
    kingpin = report_kingpin_inc(upper_bj, lower_bj)
    caster = report_caster(upper_bj, lower_bj)


    return {
        "camber": camber,
        "toe": toe,
        "kingpin_inc": kingpin,
        "caster": caster,
        "wheel_z": wheel_center[2],
        "contact_z": contact[2],
    }
