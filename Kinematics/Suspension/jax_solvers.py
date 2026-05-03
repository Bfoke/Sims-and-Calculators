import jax.numpy as jnp
import jax

'''

Chassis Coordinate System:
Origin at center of rear axle on ground plane
X - Forward
Y - Left
Z - Up

'''

'''
#######################################
# Functions for manipulating geometry #
#######################################
'''

def axis_of_rot(A: jnp.array, B: jnp.array): 
    """helper function for rotate point function"""

    vec = A - B
    mag = jnp.linalg.norm(vec)

    if mag == 0:
        raise ValueError("axis_of_rot unit vec length = 0")
    
    unit_vec = vec / mag
    return unit_vec

def rotate_point(point: jnp.array, theta: float, A: jnp.array, B: jnp.array): 
    '''rotate a point by angle theta around an axis'''

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
    # original_pos (P) and new_pos (Q) are (3, 3)
    # We want R and t such that Q = R*P + t
    P = original_pos.T 
    Q = new_pos.T

    P_centroid = jnp.mean(P, axis=1, keepdims=True)
    Q_centroid = jnp.mean(Q, axis=1, keepdims=True)

    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    # Covariance matrix
    H = Q_centered @ P_centered.T
    U, S, Vt = jnp.linalg.svd(H, full_matrices=False)

    # Reflection correction for a proper rotation matrix
    d = jnp.linalg.det(U @ Vt)
    D = jnp.diag(jnp.array([1.0, 1.0, d]))

    R = U @ D @ Vt
    t = Q_centroid - R @ P_centroid

    return R, t.flatten(), jnp.linalg.det(R), jnp.trace(R)

def lower_wb_solve_jax(u_bj, l_origin, l_axis, l_bj_0, joint_dist):
    """
    Analytical solver for lower wishbone rotation angle theta.
    Uses the rotation plane intersection to find theta without iteration.
    """
    # 1. Setup the local coordinate system for the wishbone rotation
    # P_c is the projection of l_bj_0 onto the axis (center of the rotation circle)
    v_origin_to_bj = l_bj_0 - l_origin
    dist_along_axis = jnp.dot(v_origin_to_bj, l_axis)
    P_c = l_origin + dist_along_axis * l_axis
    
    # r is the swing radius of the lower balljoint
    r = jnp.linalg.norm(l_bj_0 - P_c)
    
    # Basis vectors for the circle plane
    # u_0 is the zero-angle vector (at ride height)
    u_0 = (l_bj_0 - P_c) / (r + 1e-12)
    # v_0 is the 90-degree vector
    v_0 = jnp.cross(l_axis, u_0)
    
    # 2. Project u_bj onto the circle plane
    dist_to_plane = jnp.dot(u_bj - P_c, l_axis)
    P_proj = u_bj - dist_to_plane * l_axis
    
    # 3. Solve for theta in the plane
    # Target distance in the plane (Pythagoras: joint_dist^2 = r_eff^2 + dist_to_plane^2)
    r_eff_sq = jnp.maximum(joint_dist**2 - dist_to_plane**2, 1e-12)
    r_eff = jnp.sqrt(r_eff_sq)
    
    # Distance from circle center to projected upper BJ
    D_vec = P_proj - P_c
    D = jnp.linalg.norm(D_vec)
    
    # Law of Cosines to find the angle between D_vec and the vector to the new l_bj
    # r^2 + D^2 - 2*r*D*cos(phi) = r_eff^2
    cos_phi = (r**2 + D**2 - r_eff**2) / (2 * r * D + 1e-12)
    cos_phi = jnp.clip(cos_phi, -1.0, 1.0)
    phi = jnp.arccos(cos_phi)
    
    # Base angle of D_vec relative to u_0
    alpha = jnp.atan2(jnp.dot(D_vec, v_0), jnp.dot(D_vec, u_0))
    
    # Two possible solutions for theta (alpha +/- phi)
    theta1 = alpha + phi
    theta2 = alpha - phi
    
    # 4. Generate the 3D points using your rotation logic (or basis reconstruction)
    # We'll use the basis reconstruction for speed, which is identical to rotate_point
    l_bj1 = P_c + r * (jnp.cos(theta1) * u_0 + jnp.sin(theta1) * v_0)
    l_bj2 = P_c + r * (jnp.cos(theta2) * u_0 + jnp.sin(theta2) * v_0)
    
    # Choose the solution with lower Z
    l_bj = jnp.where(l_bj1[2] < l_bj2[2], l_bj1, l_bj2)
    
    # Calculate the actual theta relative to l_bj_0, not used but left from debugging
    final_theta = jnp.atan2(jnp.dot(l_bj - P_c, v_0), jnp.dot(l_bj - P_c, u_0))
    
    return final_theta, l_bj

def solve_toe_link_jax(
    upper_bj, lower_bj, rack_pos, 
    u_dist, l_dist, rod_len, 
    params
):
    P1, P2, P3 = jnp.asarray(upper_bj), jnp.asarray(lower_bj), jnp.asarray(rack_pos)

    # 1. Kingpin Geometry
    v_12 = P2 - P1
    d = jnp.linalg.norm(v_12)
    ex = v_12 / d
    
    # 2. Find the center and radius of the "Toe Link Circle" around the Kingpin
    x = (u_dist**2 - l_dist**2 + d**2) / (2 * d)
    P_c = P1 + x * ex
    r_sq = jnp.maximum(u_dist**2 - x**2, 1e-12)
    r = jnp.sqrt(r_sq)

    # 3. Project the Rack onto the Circle's Plane
    v_cp3 = P3 - P_c
    dist_to_plane = jnp.dot(v_cp3, ex)
    P_proj = P3 - dist_to_plane * ex  # The rack's projection onto the KP-normal plane
    
    # 4. Solve the 2D intersection in the Circle's Plane
    # Distance from circle center to projected rack
    D_vec = P_proj - P_c
    D = jnp.linalg.norm(D_vec)
    
    # Effective tie-rod length in the circle's plane
    # L_eff^2 + dist_to_plane^2 = rod_len^2
    L_eff_sq = jnp.maximum(rod_len**2 - dist_to_plane**2, 1e-12)
    L_eff = jnp.sqrt(L_eff_sq)

    # Now we just intersect two circles in the plane: 
    # Circle 1: Center (0,0), radius r
    # Circle 2: Center (D,0), radius L_eff
    # a is the distance from P_c to the chord intersection along D_vec
    a = (r**2 - L_eff**2 + D**2) / (2 * D + 1e-12) # Added epsilon for D=0
    
    h_sq = jnp.maximum(r**2 - a**2, 0.0)
    h = jnp.sqrt(h_sq + 1e-10)

    # 5. Build Basis in the Circle's Plane
    # u1 is along the line from circle center to projected rack
    u1 = D_vec / (D + 1e-12)
    # u2 is perpendicular to u1 and the kingpin axis
    u2 = jnp.cross(ex, u1)

    # 6. Final Solutions
    sol1 = P_c + a * u1 + h * u2
    sol2 = P_c + a * u1 - h * u2

    # Selection logic (same as before)
    if params["forward_rack"]:
        chosen_sol = jnp.where(sol1[0] > sol2[0], sol1, sol2)
    else:
        chosen_sol = jnp.where(sol1[0] <= sol2[0], sol1, sol2)
        
    return chosen_sol, h_sq, r_sq, 0.0, sol1, sol2
    
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
    
    toe_link, toe_sep, toe_z_sq, sol_bool, sol1, sol2 = solve_toe_link_jax(
        u_bj, 
        l_bj, 
        params['rack_origin'] + jnp.array([0, steer, 0]), 
        params['u_toe_dist'], 
        params['l_toe_dist'], 
        params['tie_rod_len'],
        params,
    )

    u_bj_0 = params['u_bj_0']
    l_bj_0 = params['l_bj_0']
    toe_0  = params['toe_link_0'] 

    P0 = jnp.stack([u_bj_0, l_bj_0, toe_0])
    P1 = jnp.stack([u_bj, l_bj, toe_link])
    
    R_upright, t_upright, determinant, trace = rigid_transform_jax(P0, P1)

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
    
    # Handle the singular case (wheel perfectly horizontal)
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
        "toe_z_sq": toe_z_sq,
        "determinant": determinant,
        "trace": trace,
        "sol_bool": sol_bool,
        "sol1": sol1,
        "sol2": sol2
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

'''
################################################
# Functions for evaluating suspension geometry #
################################################
'''

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
    """
    Find the inclination of the axle relative to the horizontal (XY) plane.
    Positive value typically indicates the 'tip' of the axle is above the 'root'.
    """
    # 1. Ensure the vector is normalized (though atan2 handles scaling, it's good practice)
    n = axle_dir / (jnp.linalg.norm(axle_dir) + 1e-12)

    # 2. Calculate the horizontal magnitude (projection onto XY plane)
    # This is the "run" in rise-over-run
    horizontal_mag = jnp.sqrt(n[0]**2 + n[1]**2)

    # 3. Calculate angle from horizontal
    # n[2] is the Z component (the "rise")
    # We use jnp.abs(horizontal_mag) to treat it as a pure inclination angle
    camber = jnp.arctan2(n[2], horizontal_mag)
    
    return camber

def report_caster(upper_bj: jnp.array, lower_bj: jnp.array):
    """Find caster angle given balljoint positions (XZ Plane)"""
    
    # Vector pointing from Lower BJ to Upper BJ
    bj_vec = upper_bj - lower_bj
    
    # In the XZ plane:
    # dx is the longitudinal lean (Forward/Rearward)
    # dz is the vertical height
    dx = bj_vec[0]
    dz = bj_vec[2]
    
    # atan2(forward, vertical) gives the angle from the vertical axis
    # We negate it depending on your convention (usually rearward lean is positive)
    caster_rad = jnp.atan2(-dx, dz) 
    
    return caster_rad

def report_kingpin_inc(upper_bj, lower_bj, side_sign):
    # Vector from Lower to Upper
    kv = upper_bj - lower_bj
    # Project into YZ plane (Front view)
    # KPI is angle of kv.y relative to kv.z
    # We multiply by side_sign so 'inboard' is always the same sign
    return jnp.arctan2(-kv[1] * side_sign, kv[2])

def report_scrub_radius(upper_bj, lower_bj, contact_point, axle_dir, side_sign):
    """
    side_sign: 1.0 for Left (+Y), -1.0 for Right (-Y)
    Positive Scrub = Steering intersection is OUTBOARD of contact patch center.
    """
    kingpin_dir = upper_bj - lower_bj
    t = (contact_point[2] - lower_bj[2]) / kingpin_dir[2]
    intersection_point = lower_bj + t * kingpin_dir
    
    # Vector from contact point to steering axis intersection
    offset_vec = contact_point - intersection_point
    offset_XY = offset_vec[:2]

    # The 'outboard' direction is the axle direction pointing away from the car
    # We ensure axle_dir_XY points outboard by using the side_sign
    axle_dir_XY = axle_dir[:2]
    outboard_unit_vec = (axle_dir_XY) / jnp.linalg.norm(axle_dir_XY)

    # Scrub is positive if the offset aligns with the outboard vector
    scrub_radius = jnp.dot(offset_XY, outboard_unit_vec)
    
    return scrub_radius

def report_mechanical_trail(upper_bj, lower_bj, contact_point, axle_dir):
    """
    Positive Trail = Intersection is IN FRONT of the contact point.
    """
    # 1. Find ground intersection
    kingpin_vec = upper_bj - lower_bj
    t = (contact_point[2] - lower_bj[2]) / kingpin_vec[2]
    intersection_point = lower_bj + t * kingpin_vec
    
    # 2. Distance vector from Contact to Intersection
    # (If intersection is forward of contact, x-component is positive)
    offset_vec = intersection_point - contact_point

    # 3. Define the tire's heading unit vector
    # We want the vector perpendicular to the axle that points FORWARD.
    # We take the axle direction and ensure the resulting X is positive.
    heading_dir = jnp.array([-axle_dir[1], axle_dir[0]])
    
    # Force the heading to be forward-facing (Positive X)
    # This removes the side-dependency of the axle_dir
    heading_unit = heading_dir / jnp.linalg.norm(heading_dir)
    heading_unit = jnp.where(heading_unit[0] < 0, -heading_unit, heading_unit)
    
    # 4. Project the offset onto the forward heading
    mechanical_trail = jnp.dot(offset_vec[:2], heading_unit)
    
    return mechanical_trail

def calculate_isa(theta: float, steer: float, params):
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
    q_f, s_f, _ = calculate_isa(theta_f, 0.0, params_f)
    q_r, s_r, _ = calculate_isa(theta_r, 0.0, params_r)
    
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

'''
###################################
# Functions for measuring results #
###################################
'''

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

    #debug values

    toe_sep = res["toe_seperation"]
    toe_z_sq = res["toe_z_sq"]
    determinant = res["determinant"]
    trace = res["trace"]
    sol_bool = res["sol_bool"]
    sol1 = res["sol1"]
    sol2 = res["sol2"]

    # instant centers
    q, s, h = calculate_isa(theta, steer, params)
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
        "contact_point": contact,
        "upper_bj": upper_bj,
        "lower_bj": lower_bj,
        "toe_link": toe_link,
        "axle_dir": axle_dir,
        "wheel_center": wheel_center,
        "R_upright": R_upright,

        #debug values

        "toe_separation": toe_sep,
        "toe_z_sq": toe_z_sq,
        "determinant": determinant,
        "trace": trace,
        "sol_bool": sol_bool,
        "sol1": sol1,
        "sol2": sol2
    }