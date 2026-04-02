import jax.numpy as jnp
import jax
import yaml

'''

Chassis Coordinate System:
Origin at center of rear axle on ground plane
X - Forward
Y - Left
Z - Up

all local positions are in chassis frame coordinates
all unlabeled positions are in world frame coordinates

'''

#######################################
# Functions for manipulating geometry #
#######################################

def axis_of_rot(A: jnp.array, B: jnp.array): 
    """helper function for rotate point function"""

    vec = A - B
    mag = jnp.linalg.norm(vec)

    if mag == 0:
        raise ValueError("axis_of_rot unit vec length = 0")
    
    unit_vec = vec / mag
    return unit_vec

def rotate_point(point: jnp.array, theta: float, A: jnp.array, B: jnp.array): 
    # move by angle theta from initial position

    axis = axis_of_rot(A, B)

    translated_point = point - B
    
    ux, uy, uz = axis
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    one_minus_cos = 1 - cos_t

    R = jnp.array([
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
    rotated_point = rotated_translated_point + B

    return rotated_point #new upright balljoint location

def rigid_transform_jax(original_pos, new_pos): 

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
        P_centroid = jnp.mean(original_pos, axis=0)
        Q_centroid = jnp.mean(new_pos, axis=0)

        # Center the point sets
        P_centered = original_pos - P_centroid
        Q_centered = new_pos - Q_centroid

        # Kabsch algorithm to find rotation
        H = jnp.dot(P_centered.T, Q_centered)
        U, S, Vt = jnp.linalg.svd(H, full_matrices = False)
        d = jnp.sign(jnp.linalg.det(jnp.dot(Vt.T, U.T)))
        D = jnp.diag(jnp.array([1, 1, d]))
        R = jnp.dot(Vt.T, jnp.dot(D, U.T))

        # Translation
        t = Q_centroid - jnp.dot(P_centroid, R)

        return R, t

def lower_wb_solve_jax(u_wb_bj, l_wb_origin, l_wb_axis, l_wb_bj_0, joint_dist):
    """
    Find lower wishbone angle/position that fits with an upper wishbone position
    """
    def get_l_bj(theta):
        # Rodrigues rotation formula
        translated = l_wb_bj_0 - l_wb_origin
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        dot = jnp.dot(l_wb_axis, translated)
        cross = jnp.cross(l_wb_axis, translated)
        rotated = translated * cos_t + cross * sin_t + l_wb_axis * dot * (1 - cos_t)
        return rotated + l_wb_origin

    def f(theta):
        l_bj = get_l_bj(theta)
        return jnp.linalg.norm(u_wb_bj - l_bj) - joint_dist

    # Newton-Raphson method
    theta = 0.0  # Initial guess
    for _ in range(10):  
        f_val, f_grad = jax.value_and_grad(f)(theta)
        theta = theta - f_val / f_grad
    
    return theta, get_l_bj(theta)

def solve_toe_link_jax(
    upper_bj,
    lower_bj,
    rack_pos,
    upper_toe_dist,
    lower_toe_dist,
    tie_rod_length,
    forward_rack=True
):
    """
    Solve for toe link position given upper balljoint, lower balljoint, and rack position.

    All inputs are 3D jax arrays in the same frame.
    Returns the physically valid toe link position.
    """

    P1 = jnp.asarray(upper_bj)
    P2 = jnp.asarray(lower_bj)
    P3 = jnp.asarray(rack_pos)

    # Basis construction
    ex = P2 - P1
    d = jnp.linalg.norm(ex)
    if d == 0:
        raise ValueError("Upper and lower ball joints coincide")

    ex /= d

    i = jnp.dot(ex, P3 - P1)
    temp = P3 - P1 - i * ex
    temp_norm = jnp.linalg.norm(temp)
    if temp_norm == 0:
        raise ValueError("Rack lies on BJ axis")

    ey = temp / temp_norm
    ez = jnp.cross(ex, ey)

    j = jnp.dot(ey, P3 - P1)

    # Coordinates in this frame
    x = (upper_toe_dist**2 - lower_toe_dist**2 + d**2) / (2 * d)
    y = (upper_toe_dist**2 - tie_rod_length**2 + i**2 + j**2 - 2 * i * x) / (2 * j)

    z_sq = upper_toe_dist**2 - x**2 - y**2

    z = jnp.sqrt(jnp.maximum(z_sq, 0.0))

    sol1 = P1 + x * ex + y * ey + z * ez
    sol2 = P1 + x * ex + y * ey - z * ez

    # Physical solution selection
    if forward_rack:
        return sol1 if sol1[0] > sol2[0] else sol2
    else:
        return sol1 if sol1[0] < sol2[0] else sol2
    
def solve_corner_jax(upper_theta: float, steer: float, params):
    """
    Differentiable suspension solver.
    Returns a dictionary of all critical 3D points and vectors.
    """
    # 1. Solve the 3 main control points (Upper BJ, Lower BJ, Toe Link)
    u_bj = rotate_point(params['u_bj_0'], upper_theta, params['u_origin'], params['u_origin'] + params['u_axis'])
    
    _, l_bj = lower_wb_solve_jax(
        u_bj, 
        params['l_origin'], 
        params['l_axis'], 
        params['l_bj_0'], 
        params['joint_dist']
    )
    
    toe_link = solve_toe_link_jax(
        u_bj, 
        l_bj, 
        params['rack_origin'] + jnp.array([0, steer, 0]), 
        params['u_toe_dist'], 
        params['l_toe_dist'], 
        params['tie_rod_len'],
        forward_rack=params['forward_rack']
    )

    # 2. Compute Rigid Transform from Initial Pose to Current Pose
    # P0: Original positions from YAML
    # P1: Current solved positions
    # u_bj_0 = params['u_bj_0']
    # l_bj_0 = params['l_bj_0']
    # toe_0  = params['toe_link_0'] 

    # P0 = jnp.stack([u_bj_0, l_bj_0, toe_0])
    # P1 = jnp.stack([u_bj, l_bj, toe_link])
    P0 = params['upright_pts_0'] # [u_bj_0, l_bj_0, toe_0]
    P1 = jnp.stack([u_bj, l_bj, toe_link])
    
    R_upright, t_upright = rigid_transform_jax(P0, P1)

    # 3. Transform Axle Geometry
    axle_root = R_upright @ params['axle_root_0'] + t_upright
    axle_tip  = R_upright @ params['axle_tip_0'] + t_upright
    
    # 4. Calculate Axle Direction and Wheel Center
    wheel_center = axle_root
    axle_vec = axle_tip - axle_root
    axle_dir = axle_vec / jnp.linalg.norm(axle_vec)

    # 5. Calculate Contact Point (Lowest point on the wheel circle)
    # The wheel is a circle in the plane normal to axle_dir.
    # We find the direction 'd' that is the projection of gravity (0,0,-1) 
    # onto the wheel plane.
    gravity = jnp.array([0.0, 0.0, -1.0])
    
    # Projection of gravity onto the wheel plane: g_proj = g - (g dot n) * n
    # n is the axle_dir
    dot_gn = jnp.dot(gravity, axle_dir)
    d_vec = gravity - dot_gn * axle_dir
    
    # Normalize the downward vector in the wheel plane
    d_norm = jnp.linalg.norm(d_vec)
    
    # Handle the singular case (wheel perfectly horizontal - unlikely in cars)
    # If d_norm is 0, we just go straight down in world Z
    unit_down_in_plane = jnp.where(d_norm > 1e-9, d_vec / d_norm, jnp.array([0.0, 0.0, -1.0]))
    
    contact_point = wheel_center + params['wheel_radius'] * unit_down_in_plane

    return {
        "upper_bj": u_bj,
        "lower_bj": l_bj,
        "toe_link": toe_link,
        "axle_dir": axle_dir,
        "wheel_center": wheel_center,
        "contact_point": contact_point,
        "R_upright": R_upright
    }

################################################
# Functions for evaluating suspension geometry #
################################################

def project_YZ(p: jnp.array):
    """Project 3D point into front view (YZ plane)."""
    return jnp.array([p[1], p[2]])  # (Y, Z)

def project_XZ(p: jnp.array):
    """Project 3D point onto side view XZ plane"""
    return jnp.array([p[0], p[2]]) # X, Z

def project_XY(p: jnp.array):
    """Project 3D point onto top view XY plane"""
    return jnp.array([p[0], p[1]]) # X, Y

def report_toe(axle_dir: jnp.array, side_sign: float):
    """find toe angle in chassis frame given axle direction vector"""
    
    n = axle_dir / jnp.linalg.norm(axle_dir)
    n_xy = project_XY(n)

    y_ref = jnp.array([0.0, 1.0 * side_sign])

    # Calculate angle
    cross = jnp.cross(y_ref, n_xy)
    dot = jnp.dot(y_ref, n_xy)

    toe = jnp.arctan2(cross, dot)

    return toe

def report_camber(axle_dir: jnp.array):

    """find camber angle given axle direction vector"""

    # report camber in chassis frame or world frame depending on which axle_dir is passed
    # TODO: add ability to pass in world frame R and t matrices so can get camber in world frame
    n = axle_dir / jnp.linalg.norm(axle_dir)

    # project into YZ plane (remove X component)
    n_yz = jnp.array([n[1], n[2]])

    # reference horizontal
    z = jnp.array([1.0, 0.0])

    camber = jnp.arctan2(n_yz[1], jnp.abs(n_yz[0]))
    return camber

def report_caster(upper_bj: jnp.array, lower_bj: jnp.array):

    """find caster angle given balljoint positions"""

    # only valid if vehicle pose is on ground and facing positive X direction
    # TODO: add chassis R and T to make it always valid
    bj_vec = upper_bj - lower_bj

    # reference vertical vector
    vertical_vec = jnp.array([0.0, 0.0, 1.0])

    # plane normal for camber: yz-plane → normal = x-axis
    plane_normal = jnp.array([1.0, 0.0, 0.0])
    plane_normal /= jnp.linalg.norm(plane_normal)   # ensure unit normal

    # project kingpin axis and vertical vector onto the plane
    bj_proj = bj_vec - jnp.dot(bj_vec, plane_normal) * plane_normal
    vert_proj = vertical_vec - jnp.dot(vertical_vec, plane_normal) * plane_normal

    # compute camber angle between the two projected vectors
    cos_theta = jnp.dot(bj_proj, vert_proj) / (
        jnp.linalg.norm(bj_proj) * jnp.linalg.norm(vert_proj)
    )

    # numerical stability
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)

    return jnp.arccos(cos_theta)

def report_kingpin_inc(upper_bj: jnp.array, lower_bj: jnp.array):

    """find kingpin inclination given upper and lower balljoint positions"""

    # kingpin axis vector
    bj_vec = upper_bj - lower_bj

    # reference vertical vector (pointing up from lower BJ)
    vertical_vec = jnp.array([0, 0, 1])

    # define plane normal: mid-plane = xz-plane → normal = y-axis
    mid_plane_normal = jnp.array([0, 1, 0], dtype=float)
    mid_plane_normal /= jnp.linalg.norm(mid_plane_normal)  # ensure unit normal

    # projection onto plane: v_proj = v - (v·n)n
    bj_proj = bj_vec - jnp.dot(bj_vec, mid_plane_normal) * mid_plane_normal
    vert_proj = vertical_vec - jnp.dot(vertical_vec, mid_plane_normal) * mid_plane_normal

    # angle between projected vectors
    cos_theta = jnp.dot(bj_proj, vert_proj) / (
        jnp.linalg.norm(bj_proj) * jnp.linalg.norm(vert_proj)
    )

    # numerical safety
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)

    return jnp.arccos(cos_theta)

def report_scrub_radius(upper_bj, lower_bj, contact_point, axle_dir, side_sign):
    """
    side_sign: 1.0 for Left (+Y), -1.0 for Right (-Y)
    Positive Scrub = Steering intersection is OUTBOARD of contact patch center.
    """
    kingpin_dir = upper_bj - lower_bj
    t = (contact_point[2] - lower_bj[2]) / kingpin_dir[2]
    intersection_point = lower_bj + t * kingpin_dir
    
    # Vector from contact point to steering axis intersection
    offset_vec = intersection_point - contact_point
    offset_XY = offset_vec[:2]

    # The 'outboard' direction is the axle direction pointing away from the car
    # We ensure axle_dir_XY points outboard by using the side_sign
    axle_dir_XY = axle_dir[:2]
    outboard_unit_vec = (axle_dir_XY * side_sign) / jnp.linalg.norm(axle_dir_XY)

    # Scrub is positive if the offset aligns with the outboard vector
    scrub_radius = jnp.dot(offset_XY, outboard_unit_vec)
    
    return scrub_radius

def report_mechanical_trail(upper_bj, lower_bj, contact_point, axle_dir):
    """
    Calculates Mechanical Trail as the distance perpendicular to the axle_dir
    (i.e., along the tire's heading) on the ground plane.
    """
    kingpin_vec = upper_bj - lower_bj
    t = (contact_point[2] - lower_bj[2]) / kingpin_vec[2]
    intersection_point = lower_bj + t * kingpin_vec
    offset_vec = intersection_point - contact_point
    
    # Heading vector is the axle vector rotated 90 degrees in XY
    # If axle is [y, -x], heading is [x, y]
    heading_unit_2d = jnp.array([-axle_dir[1], axle_dir[0]])
    heading_unit_2d /= jnp.linalg.norm(heading_unit_2d)
    
    offset_2d = jnp.array([offset_vec[0], offset_vec[1]])
    mechanical_trail = jnp.dot(offset_2d, heading_unit_2d)
    
    return mechanical_trail

def calculate_isa_exact(theta: float, steer: float, params):
    """
    Calculates the exact 3D Screw Axis using Automatic Differentiation.
    Returns: q (point on axis), s (unit direction), h (pitch)
    """
    # 1. We differentiate the solver directly
    # jac shape: (3 points, 3 coordinates)
    jac_fn = jax.jacobian(solve_corner_jax, argnums=0)
    velocities = jac_fn(theta, steer, params)
    
    # 2. Get current positions to define the rigid body state
    positions = solve_corner_jax(theta, steer, params)
    p1 = positions["upper_bj"]
    p2 = positions["lower_bj"]
    
    v1 = velocities["upper_bj"]
    v2 = velocities["lower_bj"]

    # 3. Solve for Angular Velocity vector (omega)
    r12 = p2 - p1
    v12 = v2 - v1
    omega = jnp.cross(r12, v12) / jnp.dot(r12, r12)
    
    omega_mag_sq = jnp.dot(omega, omega)
    omega_mag = jnp.sqrt(omega_mag_sq)
    
    # 4. Extract Screw Parameters
    s = omega / omega_mag
    h = jnp.dot(v1, omega) / omega_mag_sq
    q = p1 + jnp.cross(omega, v1) / omega_mag_sq

    return q, s, h

def report_roll_instant_center(q: jnp.array, s: jnp.array, wheel_x: float):
    """
    Intersection of Screw Axis with the wheel's transverse plane.
    """
    denom = s[0]
    
    # Handle axis parallel to YZ plane to avoid div by zero
    t = jnp.where(
        jnp.abs(denom) > 1e-9, 
        (wheel_x - q[0]) / denom, 
        0.0
    )

    ic_3d = q + t * s
    
    # Force X to wheel_x if it was parallel
    ic_3d = jnp.where(jnp.abs(denom) > 1e-9, ic_3d, ic_3d.at[0].set(wheel_x))

    return ic_3d

def report_pitch_instant_center(q: jnp.array, s: jnp.array, target_y: float):
    """
    Finds the intersection of the Screw Axis with a longitudinal XZ plane at target_y.
    This represents the Pitch Center for that specific corner.
    """
    # We want q_y + t * s_y = target_y
    denom = s[1] # Look at the Y component of the unit direction
    
    t = jnp.where(
        jnp.abs(denom) > 1e-9, 
        (target_y - q[1]) / denom, 
        0.0
    )

    ic_3d = q + t * s
    
    # Project to XZ coordinates
    return ic_3d # [X, Z]

def find_2d_intersection(p1, p2, p3, p4):
    """
    Finds intersection of line (p1-p2) and (p3-p4) in 2D.
    Points are jnp.array([Y, Z])
    """
    # Line 1: a1*y + b1*z = c1
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1*p1[0] + b1*p1[1]

    # Line 2: a2*y + b2*z = c2
    a2 = p4[1] - p3[1]
    b2 = p3[0] - p4[0]
    c2 = a2*p3[0] + b2*p3[1]

    determinant = a1*b2 - a2*b1
    
    # Intersection coordinates
    y = (b2*c1 - b1*c2) / determinant
    z = (a1*c2 - a2*c1) / determinant
    
    return jnp.array([y, z])

def report_roll_center(theta_r, theta_l, steer, params_r, params_l):
    """
    Solves both corners and finds the Roll Center.
    """
    # 1. Solve both corners
    data_r = solve_and_measure_corner(theta_r, steer, params_r)
    data_l = solve_and_measure_corner(theta_l, steer, params_l)
    
    # 2. Extract YZ projections for Swing Arm lines
    cp_r = project_YZ(data_r["contact_patch"])
    ic_r = project_YZ(data_r["instant_center"])
    
    cp_l = project_YZ(data_l["contact_patch"])
    ic_l = project_YZ(data_l["instant_center"])
    
    # 3. Intersection of (CP_R -> IC_R) and (CP_L -> IC_L)
    rc_yz = find_2d_intersection(cp_r, ic_r, cp_l, ic_l)
    
    return {
        "roll_center": rc_yz,
        "right_data": data_r,
        "left_data": data_l
    }

def report_pitch_center(theta_f, theta_r, params_f, params_r):

    # p sure this is wrong

    """
    Evaluates the pitch center of the vehicle in the side view (XZ).
    """
    # 1. Solve both front and rear ISA
    q_f, s_f, _ = calculate_isa_exact(theta_f, 0.0, params_f)
    q_r, s_r, _ = calculate_isa_exact(theta_r, 0.0, params_r)
    
    # 2. Get 2D Pitch ICs (SVIC) in XZ plane
    # We use the wheel center Y for each corner
    svic_f = report_pitch_instant_center(q_f, s_f, params_f['upright_pts_0'][0][1])
    svic_r = report_pitch_instant_center(q_r, s_r, params_r['upright_pts_0'][0][1])
    
    # 3. Get Contact Patches
    res_f = solve_corner_jax(theta_f, 0.0, params_f)
    res_r = solve_corner_jax(theta_r, 0.0, params_r)
    
    cp_f = project_XZ(res_f["contact_patch"])
    cp_r = project_XZ(res_r["contact_patch"])
    
    # 4. Intersect the two Swing Arm lines in XZ plane
    # Front line: CP_F -> SVIC_F
    # Rear line:  CP_R -> SVIC_R
    # Use your existing find_2d_intersection but it expects [X, Z] now
    pitch_center_xz = find_2d_intersection(cp_f, svic_f, cp_r, svic_r)
    
    return {
        "pitch_center": pitch_center_xz, # [X_coord, Z_height]
        "svic_front": svic_f,
        "svic_rear": svic_r
    }

###################################
# Functions for measuring results #
###################################

def solve_and_measure_corner(theta, steer, params):
    """
    Full kinematic evaluation of a single corner.
    """

    # solve for positions based on upper CA angle and steering/toe position
    res = solve_corner_jax(theta, steer, params) 

    # balljoint, cp, and hardpoint locations
    upper_bj = res["upper_bj"]
    lower_bj = res["lower_bj"]
    toe_link = res["toe_link"]
    contact = res["contact_point"]
    axle_dir = res["axle_dir"]
    wheel_center = res["wheel_center"]
    R_upright = res["R_upright"]
    side_sign = params["side_sign"]

    # instant centers
    q, s, h = calculate_isa_exact(theta, steer, params)
    roll_ic_3d = report_roll_instant_center(q, s, contact[0])
    pitch_ic_3d = report_pitch_instant_center(q, s, contact[1])
    # print(pitch_ic_3d)
    det = jnp.linalg.det(res["R_upright"])
    print(det)

    return {
        "camber": report_camber(axle_dir),
        "toe": report_toe(axle_dir, side_sign),
        "kingpin_inc": report_kingpin_inc(upper_bj, lower_bj),
        "caster": report_caster(upper_bj, lower_bj),
        "scrub_radius": report_scrub_radius(upper_bj, lower_bj, contact, axle_dir, side_sign),
        "mechanical_trail": report_mechanical_trail(upper_bj, lower_bj, contact, axle_dir),
        "instant_roll_center": roll_ic_3d,
        "instant_pitch_center": pitch_ic_3d,
        "isa_q": q, 
        "isa_s": s,         
        "screw_pitch": h,
        "wheel_z": wheel_center[2],
        "contact_patch": contact,
        "upper_bj": upper_bj,
        "lower_bj": lower_bj,
        "toe_link": toe_link,
        "axle_dir": axle_dir,
        "wheel_center": wheel_center,
        "R_upright": R_upright
    }