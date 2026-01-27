import numpy as np

'''

Chassis Coordinate System:
Origin at center of rear axle on ground plane
X - Forward
Y - Left
Z - Up

all local positions are in chassis frame coordinates
all unlabeled positions are in world frame coordinates

'''

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
    
    def set_balljoint_pos(self, theta): 
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
    
    # TODO: add a function later to return steering wheel theta as function of rack travel using range and rotations
    # input: current right or left tie rod pos
    # output: wheel theta

class Upright:
    def __init__(self, upper_balljoint, lower_balljoint, toe_link, axle_root, axle_tip, wheel_rad):
        # roots are points along the vector between upper and lower balljoint that the orthogonal projection of toe_link and axle tip go to
        # they are used to locate the toe_link and axle_tip relative to the balljoint vector

        #0 positions of joints
        self.upper_balljoint_0 = np.copy(upper_balljoint)
        self.lower_balljoint_0 = np.copy(lower_balljoint)
        self.toe_link_0 = np.copy(toe_link)

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
    
class Chassis:
    def __init__(self, hardpoints):
        # TODO: maybe add cg point to help tilt decisions from non-coplanar tire positions
        
        self.hardpoints = hardpoints  # dict of chassis frame points
        self.R = np.eye(3)
        self.t = np.zeros(3)

    def to_world(self, p_local):
        return self.R @ p_local + self.t

    def set_pose(self, R, t):
        self.R = R
        self.t = t

class Corner:
    def __init__(self, upper_wb: Wishbone, lower_wb: Wishbone, upright: Upright, wheel: Wheel, rack: Rack, side):

        self.u_wb = upper_wb #wishbone object
        self.l_wb = lower_wb #wishbone object
        self.upright = upright #upright object
        self.rack = rack #object, steering rack rod ends if front or toe link locations if rear
        self.side = side # 0 for left, 1 for right
        self.wheel = wheel # wheel object