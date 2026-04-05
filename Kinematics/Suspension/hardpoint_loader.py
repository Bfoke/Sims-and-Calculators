import yaml
import jax.numpy as jnp

def load_suspension_params(file_path, corner_name="front_right"):
    with open(file_path, 'r') as f:
        all_data = yaml.safe_load(f)
    
    is_left = "left" in corner_name.lower()
    
    source_name = corner_name
    if is_left and corner_name not in all_data:
        source_name = corner_name.replace("left", "right")
    
    data = all_data[source_name]
    up = data['upright']
    steer = data['steering']

    def to_jnp(list_val):
        arr = jnp.array(list_val, dtype=jnp.float32)
        if is_left:
            arr = arr.at[1].multiply(-1.0)
        return arr

    # --- Points ---
    u_f = to_jnp(data['upper_wishbone']['front'])
    u_r = to_jnp(data['upper_wishbone']['rear'])
    u_bj_0 = to_jnp(data['upper_wishbone']['balljoint_0'])
    
    l_f = to_jnp(data['lower_wishbone']['front'])
    l_r = to_jnp(data['lower_wishbone']['rear'])
    l_bj_0 = to_jnp(data['lower_wishbone']['balljoint_0'])
    
    rack_origin = to_jnp(steer['rack_end_0'])
    toe_link_0 = to_jnp(up['toe_link_0'])

    # --- Side Sign Logic ---
    base_side_sign = data.get('side_sign', -1.0)
    current_side_sign = -base_side_sign if is_left else base_side_sign

    params = {
        # chassis side pickup points
        "u_front": u_f,
        "u_rear": u_r,
        "l_front": l_f,
        "l_rear": l_r,
        
        #upper wishbone
        "u_origin": u_r,
        "u_axis": (u_f - u_r) / jnp.linalg.norm(u_f - u_r),
        "u_bj_0": u_bj_0,

        # lower wishbone
        "l_origin": l_r,
        "l_axis": (l_f - l_r) / jnp.linalg.norm(l_f - l_r),
        "l_bj_0": l_bj_0,

        # distances between upright joints
        "joint_dist": jnp.linalg.norm(u_bj_0 - l_bj_0),
        "u_toe_dist": jnp.linalg.norm(u_bj_0 - toe_link_0),
        "l_toe_dist": jnp.linalg.norm(l_bj_0 - toe_link_0),
        
        # steering/ toe
        "rack_origin": rack_origin,
        "tie_rod_len": jnp.linalg.norm(toe_link_0 - rack_origin),
        "forward_rack": steer.get('forward_rack', True), 
        "toe_link_0": toe_link_0,

        "upright_pts_0": jnp.stack([
            u_bj_0, 
            l_bj_0, 
            toe_link_0
        ]),
        "axle_root_0": to_jnp(up['axle_root_0']),
        "axle_tip_0": to_jnp(up['axle_tip_0']),
        "wheel_radius": jnp.array(up['wheel_radius']),
        "side_sign": current_side_sign
    }
    
    return params