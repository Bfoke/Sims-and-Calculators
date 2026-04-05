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
    # original_pos and new_pos are (3, 3) -> 3 points, each with 3 coords
    P = original_pos.T # Make it (3, N)
    Q = new_pos.T

    P_centroid = jnp.mean(P, axis=1, keepdims=True)
    Q_centroid = jnp.mean(Q, axis=1, keepdims=True)

    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    H = Q_centered @ P_centered.T
    U, S, Vt = jnp.linalg.svd(H, full_matrices = False)
    
    R = U @ Vt
    # Reflection handle
    d = jnp.linalg.det(R)
    U = U.at[:, 2].multiply(jnp.where(d < 0, -1.0, 1.0))
    R = U @ Vt

    t = Q_centroid - R @ P_centroid

    return R, t.flatten()

# def lower_wb_solve_jax_old(u_wb_bj, l_wb_origin, l_wb_axis, l_wb_bj_0, joint_dist):
#     """
#     Find lower wishbone angle/position that fits with an upper wishbone position
#     """
#     def get_l_bj(theta):
#         # Rodrigues rotation formula
#         translated = l_wb_bj_0 - l_wb_origin
#         cos_t = jnp.cos(theta)
#         sin_t = jnp.sin(theta)
#         dot = jnp.dot(l_wb_axis, translated)
#         cross = jnp.cross(l_wb_axis, translated)
#         rotated = translated * cos_t + cross * sin_t + l_wb_axis * dot * (1 - cos_t)
#         return rotated + l_wb_origin

#     def f(theta):
#         l_bj = get_l_bj(theta)
#         return jnp.linalg.norm(u_wb_bj - l_bj) - joint_dist

#     # Newton-Raphson method
#     theta = 0.0  # Initial guess
#     for _ in range(10):  
#         f_val, f_grad = jax.value_and_grad(f)(theta)
#         theta = theta - f_val / f_grad
    
#     return theta, get_l_bj(theta)

def lower_wb_solve_jax(u_wb_bj, l_wb_origin, l_wb_axis, l_wb_bj_0, joint_dist):
    """
    Find lower wishbone angle with a penalty to keep it below the upper BJ.
    """
    def get_l_bj(theta):
        translated = l_wb_bj_0 - l_wb_origin
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        dot = jnp.dot(l_wb_axis, translated)
        cross = jnp.cross(l_wb_axis, translated)
        rotated = translated * cos_t + cross * sin_t + l_wb_axis * dot * (1 - cos_t)
        return rotated + l_wb_origin

    def f(theta):
        l_bj = get_l_bj(theta)
        
        # 1. Standard kinematic constraint (Distance)
        dist_err = jnp.linalg.norm(u_wb_bj - l_bj) - joint_dist
        
        # 2. Height Penalty: Lower BJ must be below Upper BJ (l_bj[2] < u_wb_bj[2])
        # If height_diff is positive (inverted), we add a massive penalty.
        height_diff = l_bj[2] - u_wb_bj[2]
        
        # We use a soft-plus style penalty so the gradient points "down" 
        # even if it accidentally crosses over.
        # Penalty = (height_diff + 0.1)^2 if height_diff > -0.05
        penalty = jnp.where(height_diff > -0.05, 1000.0 * (height_diff + 0.05)**2, 0.0)
        
        return dist_err**2 + penalty # Minimize squared error + penalty

    # 1. Better Initial Guess: Try a few angles to find the "low" side
    # Newton-Raphson is a local optimizer; it needs to start in the right valley.
    search_range = jnp.linspace(-jnp.pi/3, jnp.pi/3, 8)
    costs = jax.vmap(f)(search_range)
    theta = search_range[jnp.argmin(costs)]

    # 2. Newton-Raphson Refinement
    for _ in range(10):  
        f_val, f_grad = jax.value_and_grad(f)(theta)
        # Add epsilon to grad to avoid div by zero
        theta = theta - f_val / (f_grad + 1e-9)
    
    return theta, get_l_bj(theta)

def solve_toe_link_jax_old(
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
        # If rack is forward, we usually want the solution with the larger X 
        # (further forward). jnp.where handles the tracer correctly.
        return jnp.where(sol1[0] > sol2[0], sol1, sol2)
    else:
        # If rack is rearward, we want the smaller X
        return jnp.where(sol1[0] < sol2[0], sol1, sol2)

# def solve_toe_link_jax(
#     upper_bj, lower_bj, rack_pos, 
#     upper_toe_dist, lower_toe_dist, tie_rod_length, 
#     params, forward_rack=True
# ):
# #has discontinuity at theta=0 but values seem mostly correct
#     P1, P2, P3 = jnp.asarray(upper_bj), jnp.asarray(lower_bj), jnp.asarray(rack_pos)

#     ex = (P2 - P1) / jnp.linalg.norm(P2 - P1)
#     i = jnp.dot(ex, P3 - P1)
#     temp = P3 - P1 - i * ex
#     ey = temp / jnp.linalg.norm(temp)
#     ez = jnp.cross(ex, ey)
#     j = jnp.dot(ey, P3 - P1)
#     d = jnp.linalg.norm(P2 - P1)

#     x = (upper_toe_dist**2 - lower_toe_dist**2 + d**2) / (2 * d)
#     y = (upper_toe_dist**2 - tie_rod_length**2 + i**2 + j**2 - 2 * i * x) / (2 * j)
    
#     z_sq = upper_toe_dist**2 - x**2 - y**2
#     z = jnp.sqrt(jnp.maximum(z_sq, 0.0))

#     sol1 = P1 + x * ex + y * ey + z * ez
#     sol2 = P1 + x * ex + y * ey - z * ez

#     diff1 = jnp.linalg.norm(sol1 - params['toe_link_0'])
#     diff2 = jnp.linalg.norm(sol2 - params['toe_link_0'])
#     chosen_sol = jnp.where(diff1 < diff2, sol1, sol2)
    
#     # Diagnostics: 
#     # 1. Separation is the distance between the two potential solutions (2 * z)
#     # 2. z_sq shows how "deep" into the valid geometry we are (if negative, geometry is impossible)
#     separation = jnp.linalg.norm(sol1 - sol2)
    
#     return chosen_sol, separation, z_sq

def solve_toe_link_jax(
    upper_bj, lower_bj, rack_pos, 
    upper_toe_dist, lower_toe_dist, tie_rod_length, 
    params, forward_rack=True
):
    P1, P2, P3 = jnp.asarray(upper_bj), jnp.asarray(lower_bj), jnp.asarray(rack_pos)

    # 1. Kingpin Axis (Lower to Upper)
    bj_vec = P2 - P1
    d = jnp.linalg.norm(bj_vec)
    ex = bj_vec / d

    # 2. STABLE BASIS: Use Global X (Forward) as reference
    # This is extremely stable for car geometry and won't flip at theta=0
    ref = jnp.array([1.0, 0.0, 0.0])
    ey_vec = jnp.cross(ex, ref)
    ey = ey_vec / (jnp.linalg.norm(ey_vec) + 1e-9)
    ez = jnp.cross(ex, ey)

    # 3. Project Rack (P3) into this local frame
    p3_rel = P3 - P1
    i = jnp.dot(p3_rel, ex)
    j = jnp.dot(p3_rel, ey)
    k = jnp.dot(p3_rel, ez)

    # 4. MATH: Intersection of 3 Spheres
    # Sphere 1 & 2 intersect to form a circle in the ey-ez plane
    x = (upper_toe_dist**2 - lower_toe_dist**2 + d**2) / (2 * d)
    r_sq = upper_toe_dist**2 - x**2
    r = jnp.sqrt(jnp.maximum(r_sq, 0.0))

    # Now intersect that circle (radius r) with Sphere 3 (Rack)
    # We solve this in the 2D plane of ey-ez
    h_sq = tie_rod_length**2 - (x - i)**2
    h = jnp.sqrt(jnp.maximum(h_sq, 0.0))
    
    # Distance from BJ-axis to Rack in the ey-ez plane
    d_plane = jnp.sqrt(jnp.maximum(j**2 + k**2, 1e-9))
    
    # Standard 2D circle-circle intersection
    a = (r**2 - h**2 + d_plane**2) / (2 * d_plane)
    h_chord = jnp.sqrt(jnp.maximum(r**2 - a**2, 0.0))
    
    # Base point along the line from BJ-axis to Rack projection
    y_base = (a / d_plane) * j
    z_base = (a / d_plane) * k
    
    # Two potential solutions (offset perpendicular to the j-k vector)
    sol1_y = y_base + (h_chord / d_plane) * k
    sol1_z = z_base - (h_chord / d_plane) * j
    
    sol2_y = y_base - (h_chord / d_plane) * k
    sol2_z = z_base + (h_chord / d_plane) * j

    # 5. Transform back to World Space
    sol1 = P1 + x * ex + sol1_y * ey + sol1_z * ez
    sol2 = P1 + x * ex + sol2_y * ey + sol2_z * ez

    # 6. Selection based on distance to static YAML point
    diff1 = jnp.linalg.norm(sol1 - params['toe_link_0'])
    diff2 = jnp.linalg.norm(sol2 - params['toe_link_0'])
    chosen_sol = jnp.where(diff1 < diff2, sol1, sol2)
    
    separation = jnp.linalg.norm(sol1 - sol2)
    # print(chosen_sol)
    return chosen_sol, separation, r_sq
    
    
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
    
    toe_link, toe_sep, toe_z_sq = solve_toe_link_jax(
        u_bj, 
        l_bj, 
        params['rack_origin'] + jnp.array([0, steer, 0]), 
        params['u_toe_dist'], 
        params['l_toe_dist'], 
        params['tie_rod_len'],
        params,
        forward_rack=params['forward_rack']
    )

    # 2. Compute Rigid Transform from Initial Pose to Current Pose
    # P0: Original positions from hardpoints YAML
    # P1: Current solved positions

    u_bj_0 = params['u_bj_0']
    l_bj_0 = params['l_bj_0']
    toe_0  = params['toe_link_0'] 

    P0 = jnp.stack([u_bj_0, l_bj_0, toe_0])
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
    gravity = jnp.array([0.0, 0.0, -1.0])
    
    # Projection of gravity onto the wheel plane: g_proj = g - (g dot n) * n
    # n is the axle_dir
    dot_gn = jnp.dot(gravity, axle_dir)
    d_vec = gravity - dot_gn * axle_dir
    
    # Normalize the downward vector in the wheel plane
    d_norm = jnp.linalg.norm(d_vec)
    
    # Handle the singular case (wheel perfectly horizontal, you fucked up)
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
        "R_upright": R_upright,
        "toe_seperation": toe_sep,
        "toe_z_sq": toe_z_sq
    }

def solve_theta_for_ground(steer, world_params):
    def objective(theta):
        res = solve_corner_jax(theta, steer, world_params)
        return res["contact_point"][2] 

    # Brute force search (This is safe for tracing because search_range is static)
    search_range = jnp.linspace(-jnp.pi/4, jnp.pi/4, 20)
    z_vals = jax.vmap(objective)(search_range)
    
    # Use jnp.argmin to find the starting point
    best_idx = jnp.argmin(jnp.abs(z_vals))
    theta_guess = search_range[best_idx]

    # Newton-Raphson
    t = theta_guess
    for _ in range(10):
        val, grad = jax.value_and_grad(objective)(t)
        # Use a small epsilon to prevent NaN if grad is 0
        t = t - val / (grad + 1e-9)
        
    return t

def update_full_car(chassis_pose, steering_input, all_params):
    """
    chassis_pose: {'xyz': [x,y,z], 'roll': r, 'pitch': p}
    """
    world_results = {}
    
    for side in ["fr", "fl", "rr", "rl"]:
        # 1. Move chassis points to world space
        world_p = get_world_params(
            all_params[side], 
            chassis_pose['xyz'], 
            chassis_pose['roll'], 
            chassis_pose['pitch']
        )
        
        # 2. Find the theta that keeps the tire on the ground
        target_theta = solve_theta_for_ground(steering_input, world_p)
        
        # 3. Solve the final kinematics for visualization
        world_results[side] = solve_and_measure_corner(target_theta, steering_input, world_p)
        
    return world_results

def get_world_params(local_params, chassis_xyz, roll, pitch):
    """
    Transforms local chassis hardpoints into World Space.
    chassis_xyz: [x, y, z] position of the YAML origin in the world.
    """
    # Rotation Matrices (Roll then Pitch)
    cR, sR = jnp.cos(roll), jnp.sin(roll)
    cP, sP = jnp.cos(pitch), jnp.sin(pitch)
    
    # Roll (X-axis)
    Rx = jnp.array([[1, 0, 0], [0, cR, -sR], [0, sR, cR]])
    # Pitch (Y-axis)
    Ry = jnp.array([[cP, 0, sP], [0, 1, 0], [-sP, 0, cP]])
    
    R_chassis = Ry @ Rx
    
    def transform_pt(pt):
        return R_chassis @ jnp.asarray(pt) + jnp.asarray(chassis_xyz)

    # Return a new dict with world coordinates
    world_p = dict(local_params)
    for key in ['u_front', 'u_rear', 'l_front', 'l_rear', 'rack_origin', 
                'u_bj_0', 'l_bj_0', 'toe_link_0', 'axle_root_0', 'axle_tip_0']:
        world_p[key] = transform_pt(local_params[key])
    
    # Vectors (directions) only get rotated, not translated
    world_p['u_axis'] = R_chassis @ local_params['u_axis']
    world_p['l_axis'] = R_chassis @ local_params['l_axis']
    
    # Update the upright_pts_0 stack for the rigid transform
    world_p['upright_pts_0'] = jnp.stack([world_p['u_bj_0'], world_p['l_bj_0'], world_p['toe_link_0']])
    
    return world_p

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
    bj_vec = upper_bj - lower_bj

    # reference vertical vector
    vertical_vec = jnp.array([0.0, 0.0, 1.0])

    # plane normal for caster: xz-plane → normal = y-axis
    plane_normal = jnp.array([0.0, 1.0, 0.0])
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

def report_kingpin_inc_has_jump_bug(upper_bj: jnp.array, lower_bj: jnp.array):

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

def report_kingpin_inc(upper_bj: jnp.array, lower_bj: jnp.array, side_sign: float):
    """
    Find kingpin inclination (KPI) given balljoint positions.
    side_sign: 1.0 for Left (+Y), -1.0 for Right (-Y)
    """
    # 1. Kingpin vector (Lower to Upper)
    bj_vec = upper_bj - lower_bj
    
    # 2. Reference vertical vector
    vertical_vec = jnp.array([0.0, 0.0, 1.0])

    # 3. Define the Front-View Plane (YZ-plane)
    # The normal to this plane is the Global X (Forward) axis
    plane_normal = jnp.array([1.0, 0.0, 0.0])

    # 4. Project the Kingpin vector onto the YZ-plane
    # v_proj = v - (v·n)n
    bj_proj = bj_vec - jnp.dot(bj_vec, plane_normal) * plane_normal
    
    # 5. Calculate angle using arctan2 for 360-degree stability
    # We want the angle between the vertical and the projected BJ vector
    # We multiply the Y-component by side_sign so that 'inboard' is always the same direction
    y_comp = bj_proj[1] * side_sign
    z_comp = bj_proj[2]
    
    # KPI is the angle from the vertical (Z-axis) toward the centerline
    # arctan2(y, z) gives the angle relative to the Z-axis
    kpi = jnp.arctan2(-y_comp, z_comp)

    return kpi

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
    Intersection of Screw Axis with the wheel's transverse plane (YZ plane of chassis).
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
    toe_sep = res["toe_seperation"]
    toe_z_sq = res["toe_z_sq"]

    # instant centers
    q, s, h = calculate_isa_exact(theta, steer, params)
    roll_ic_3d = report_roll_instant_center(q, s, contact[0])
    pitch_ic_3d = report_pitch_instant_center(q, s, contact[1])

    return {
        "camber": report_camber(axle_dir),
        "toe": report_toe(axle_dir, side_sign),
        "kingpin_inc": report_kingpin_inc(upper_bj, lower_bj, side_sign),
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
        "R_upright": R_upright,
        "toe_separation": toe_sep,
        "toe_z_sq": toe_z_sq
    }