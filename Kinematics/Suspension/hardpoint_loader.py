import yaml
import jax.numpy as jnp

def load_suspension_params(file_path, corner_name="front_right"):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)[corner_name]

    def to_jnp(list_val):
        return jnp.array(list_val, dtype=jnp.float32)

    # Pre-calculate derived constants to save JAX ops
    u_f = to_jnp(data['upper_wishbone']['front'])
    u_r = to_jnp(data['upper_wishbone']['rear'])
    u_axis = u_f - u_r
    
    l_f = to_jnp(data['lower_wishbone']['front'])
    l_r = to_jnp(data['lower_wishbone']['rear'])
    l_axis = l_f - l_r

    up = data['upright']
    
    params = {
        # Upper WB
        "u_origin": u_r,
        "u_axis": u_axis / jnp.linalg.norm(u_axis),
        "u_bj_0": to_jnp(data['upper_wishbone']['balljoint_0']),

        # Lower WB
        "l_origin": l_r,
        "l_axis": l_axis / jnp.linalg.norm(l_axis),
        "l_bj_0": to_jnp(data['lower_wishbone']['balljoint_0']),

        # Distances (Fixed constraints)
        "joint_dist": jnp.linalg.norm(to_jnp(up['upper_balljoint_0']) - to_jnp(up['lower_balljoint_0'])),
        "u_toe_dist": jnp.linalg.norm(to_jnp(up['upper_balljoint_0']) - to_jnp(up['toe_link_0'])),
        "l_toe_dist": jnp.linalg.norm(to_jnp(up['lower_balljoint_0']) - to_jnp(up['toe_link_0'])),
        
        # Steering
        "rack_origin": to_jnp(data['steering']['rack_end_0']),
        "tie_rod_len": jnp.array(data['steering']['tie_rod_length']),

        # Upright geometry for coordinate transformation
        "upright_pts_0": jnp.stack([
            to_jnp(up['upper_balljoint_0']),
            to_jnp(up['lower_balljoint_0']),
            to_jnp(up['toe_link_0'])
        ]),
        "axle_root_0": to_jnp(up['axle_root_0']),
        "axle_tip_0": to_jnp(up['axle_tip_0']),
        "wheel_radius": jnp.array(up['wheel_radius'])
    }
    
    return params