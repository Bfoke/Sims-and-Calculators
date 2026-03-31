import yaml
import jax.numpy as jnp

def load_suspension_params(file_path, corner_name="front_right"):
    with open(file_path, 'r') as f:
        all_data = yaml.safe_load(f)
    
    # Determine side based on name
    is_left = "left" in corner_name.lower()
    
    # If the exact name isn't in YAML (e.g., front_left), 
    # look for its right-side counterpart to mirror
    source_name = corner_name
    if is_left and corner_name not in all_data:
        source_name = corner_name.replace("left", "right")
    
    data = all_data[source_name]

    def to_jnp(list_val):
        arr = jnp.array(list_val, dtype=jnp.float32)
        if is_left:
            # Flip Y coordinate (index 1)
            arr = arr.at[1].multiply(-1.0)
        return arr

    # --- Pre-calculate derived constants ---
    u_f = to_jnp(data['upper_wishbone']['front'])
    u_r = to_jnp(data['upper_wishbone']['rear'])
    # Bj needs mirroring too
    u_bj_0 = to_jnp(data['upper_wishbone']['balljoint_0'])
    
    l_f = to_jnp(data['lower_wishbone']['front'])
    l_r = to_jnp(data['lower_wishbone']['rear'])
    l_bj_0 = to_jnp(data['lower_wishbone']['balljoint_0'])

    up = data['upright']
    
    # Mirror the side_sign: if Right is -1.0, Left is 1.0
    # Use the value from YAML if it exists, otherwise default to -1.0 for Right
    base_side_sign = data.get('side_sign', -1.0)
    current_side_sign = -base_side_sign if is_left else base_side_sign

    params = {
        "u_front": u_f,
        "u_rear": u_r,
        "l_front": l_f,
        "l_rear": l_r,
        
        # Upper WB (Origin is rear pickup)
        "u_origin": u_r,
        "u_axis": (u_f - u_r) / jnp.linalg.norm(u_f - u_r),
        "u_bj_0": u_bj_0,

        # Lower WB
        "l_origin": l_r,
        "l_axis": (l_f - l_r) / jnp.linalg.norm(l_f - l_r),
        "l_bj_0": l_bj_0,

        # Distances (Fixed constraints - scalars don't need mirroring)
        "joint_dist": jnp.linalg.norm(to_jnp(up['upper_balljoint_0']) - to_jnp(up['lower_balljoint_0'])),
        "u_toe_dist": jnp.linalg.norm(to_jnp(up['upper_balljoint_0']) - to_jnp(up['toe_link_0'])),
        "l_toe_dist": jnp.linalg.norm(to_jnp(up['lower_balljoint_0']) - to_jnp(up['toe_link_0'])),
        
        # Steering
        "rack_origin": to_jnp(data['steering']['rack_end_0']),
        "tie_rod_len": jnp.array(data['steering']['tie_rod_length']),

        # Upright points for rigid transform
        "upright_pts_0": jnp.stack([
            to_jnp(up['upper_balljoint_0']),
            to_jnp(up['lower_balljoint_0']),
            to_jnp(up['toe_link_0'])
        ]),
        "axle_root_0": to_jnp(up['axle_root_0']),
        "axle_tip_0": to_jnp(up['axle_tip_0']),
        "wheel_radius": jnp.array(up['wheel_radius']),
        "side_sign": current_side_sign
    }
    
    return params