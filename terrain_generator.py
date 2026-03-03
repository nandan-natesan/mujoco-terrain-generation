import xml.etree.ElementTree as xml_et
import numpy as np
import cv2
import noise
import os
import math

def euler_to_quat(roll, pitch, yaw):
    """
    Convert ZYX Euler angles to a quaternion [w, x, y, z].

    @param roll  rotation around x-axis [rad]
    @param pitch rotation around y-axis [rad]
    @param yaw   rotation around z-axis [rad]
    @return np.ndarray (4,) quaternion [w, x, y, z]
    """
    
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


def euler_to_rot(roll, pitch, yaw):
    """
    Convert ZYX Euler angles to a 3x3 rotation matrix.

    @param roll  rotation around x-axis [rad]
    @param pitch rotation around y-axis [rad]
    @param yaw   rotation around z-axis [rad]
    @return np.ndarray (3, 3) rotation matrix
    """
    
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )

    rot_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x



def rot2d(x, y, yaw):
    """
    Rotate a 2D point (x, y) by angle yaw.

    @param x    x-coordinate
    @param y    y-coordinate
    @param yaw  rotation angle [rad]
    @return (float, float) rotated (x, y)
    """
    nx = x * np.cos(yaw) - y * np.sin(yaw)
    ny = x * np.sin(yaw) + y * np.cos(yaw)
    return nx, ny



def rot3d(pos, euler):
    """
    Rotate a 3D vector by ZYX Euler angles.

    @param pos   np.ndarray (3,) vector
    @param euler np.ndarray (3,) [roll, pitch, yaw]
    @return np.ndarray (3,) rotated vector
    """
    R = euler_to_rot(euler[0], euler[1], euler[2])
    return R @ pos


def list_to_str(vec):
    return " ".join(str(s) for s in vec)


class TerrainGenerator:

    def __init__(self, input_scene_path, output_scene_path) -> None:
        self.input_scene_path=input_scene_path
        self.output_scene_path = output_scene_path
        
        self.scene = xml_et.parse(input_scene_path)
        self.root = self.scene.getroot()
        self.worldbody = self.root.find("worldbody")
        self.asset = self.root.find("asset")

    def AddBox(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1, 0.1]):
        """Add a box geometry to the scene."""
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # MuJoCo uses half-extents for box size
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
    
    def AddGeometry(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1],
               geo_type="box"):  # "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
        """Add a generic geometry primitive to the scene."""
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size for MuJoCo
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddStairs(self,
                  init_pos=[1.0, 0.0, 0.0],
                  yaw=0.0,
                  width=0.2,
                  height=0.15,
                  length=1.5,
                  stair_nums=10):
        '''
        Adds a staircase of box geometries.

        @param init_pos     [start_x, start_y, start_z] position of the first stair
        @param yaw          rotation of stairs around z axis [rad]
        @param width        width of each stair along the x axis
        @param height       height of each stair along the z axis
        @param length       length of each stair along the y axis
        @param stair_nums   number of stairs
        '''
        x, y, z = init_pos
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        
        for i in range(stair_nums):
            dx = i * (length/(length*50)) * cos_yaw
            dy = i * width * sin_yaw
            dz = i * height
            stair_pos = [x + dx, y + dy, z + dz]
            stair_size = [width/2, length/2, height/2]
            self.AddBox(position=stair_pos, size=stair_size, euler=[0,0,yaw])

    def AddRoughGround(self,
                       init_pos=[1.0, 0.0, 0.0],
                       euler=[0.0, -0.0, 0.0],
                       nums=[10, 10],
                       box_size=[0.5, 0.5, 0.5],
                       box_euler=[0.0, 0.0, 0.0],
                       separation=[0.2, 0.2],
                       box_size_rand=[0.05, 0.05, 0.05],
                       box_euler_rand=[0.2, 0.2, 0.2],
                       separation_rand=[0.05, 0.05]):
        '''
        Adds a grid of randomized box obstacles for rough terrain.

        @param init_pos         [start_x, start_y, start_z] position of the first box
        @param euler            rotation of the whole rough ground around x, y, z axis
        @param nums             [num_x, num_y] number of boxes along x and y axis
        @param box_size         [size_x, size_y, size_z] size of each box
        @param box_euler        [roll, pitch, yaw] rotation of each box around x, y, z axis
        @param separation       [sep_x, sep_y] separation between boxes along x and y axis
        @param box_size_rand    [rand_x, rand_y, rand_z] random range of box size
        @param box_euler_rand   [rand_roll, rand_pitch, rand_yaw] random range of box euler
        @param separation_rand  [rand_sep_x, rand_sep_y] random range of separation
        '''
        num_x, num_y = nums
    
        for ix in range(num_x):
          for iy in range(num_y):
            # Base position for each box in the grid
            x = ix * (box_size[0] + separation[0])
            y = iy * (box_size[1] + separation[1])
            z = box_size[2] / 2.0  # center of box

            # Add random variations to size for natural-looking terrain
            rx = np.random.uniform(-box_size_rand[0], box_size_rand[0])
            ry = np.random.uniform(-box_size_rand[1], box_size_rand[1])
            rz = np.random.uniform(-box_size_rand[2], box_size_rand[2])
            size = [box_size[0] + rx, box_size[1] + ry, box_size[2] + rz]
            
            # Random Euler rotation per box
            r_roll = box_euler[0] + np.random.uniform(-box_euler_rand[0], box_euler_rand[0])
            r_pitch = box_euler[1] + np.random.uniform(-box_euler_rand[1], box_euler_rand[1])
            r_yaw = box_euler[2] + np.random.uniform(-box_euler_rand[2], box_euler_rand[2])
            
            # Random separation offset for irregular spacing
            sep_x = np.random.uniform(-separation_rand[0], separation_rand[0])
            sep_y = np.random.uniform(-separation_rand[1], separation_rand[1])
            
            # Local position, then rotate by overall euler and translate to init_pos
            local_pos = np.array([x + sep_x, y + sep_y, z])
            rotated_pos = rot3d(local_pos, euler)
            pos = [rotated_pos[0] + init_pos[0],
                   rotated_pos[1] + init_pos[1],
                   rotated_pos[2] + init_pos[2]]
            
            self.AddBox(position=pos, euler=[r_roll, r_pitch, r_yaw], size=size)

    def AddPerlinHeighField(
        self,
        position=[1.0, 0.0, 0.0],  # position
        euler=[0.0, -0.0, 0.0],  # attitude
        size=[1.0, 1.0],  # width and length
        height_scale=0.2,  # max height
        negative_height=0.2,  # height in the negative direction of z axis
        image_width=128,  # height field image size
        img_height=128,
        smooth=100.0,  # smooth scale
        perlin_octaves=6,  # perlin noise parameter
        perlin_persistence=0.5,
        perlin_lacunarity=2.0,
        output_hfield_image="height_field.png"):
        '''
        Generates a height field using Perlin noise and adds it to the scene.
        Creates a grayscale image and registers it as a MuJoCo hfield asset.
        '''
        # Create empty terrain image
        terrain_image = np.zeros((img_height, image_width), dtype=np.float32)

        # Generate Perlin noise for natural-looking elevation
        for i in range(img_height):
            for j in range(image_width):
                x = j / smooth
                y = i / smooth
                terrain_image[i, j] = noise.pnoise2(x, y,
                                                    octaves=perlin_octaves,
                                                    persistence=perlin_persistence,
                                                    lacunarity=perlin_lacunarity,
                                                    repeatx=image_width,
                                                    repeaty=img_height,
                                                    base=0)

        # Normalize noise to 0-255 for grayscale image
        terrain_image = (terrain_image - terrain_image.min()) / (terrain_image.max() - terrain_image.min())
        terrain_image = (terrain_image * 255).astype(np.uint8)

        # Save heightfield image next to the output XML
        hfield_save_file = os.path.join(os.path.dirname(self.output_scene_path), output_hfield_image)
        cv2.imwrite(hfield_save_file, terrain_image)

        # Add hfield asset to the scene
        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        hfield.attrib["size"] = list_to_str([size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = "./" + output_hfield_image

        # Add hfield geom to worldbody
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "perlin_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddHeighFieldFromImage(
            self,
            position=[1.0, 0.0, 0.0],
            euler=[0.0, -0.0, 0.0],
            size=[2.0, 1.6],
            height_scale=0.02,
            negative_height=0.1,
            input_img=None,
            output_hfield_image="height_field.png",
            image_scale=[1.0, 1.0],
            invert_gray=False):
        '''
        Load a height field from an input image. Override or extend for custom heightmaps.
        '''
        raise NotImplementedError("AddHeighFieldFromImage: use AddPerlinHeighField or implement custom logic")

    def CustomTerrain(self):
        '''
        Override this method to add your own custom procedural terrain components.
        '''
        pass

    def Save(self):
        """Write the modified scene to the output XML path."""
        self.scene.write(self.output_scene_path)


if __name__ == "__main__":
    input_scene_path = "./google_barkour_vb/scene_mjx.xml"
    output_scene_path = "./google_barkour_vb/scene_mjx_with_terrain.xml"
    tg = TerrainGenerator(input_scene_path, output_scene_path)

    # Box obstacle
    tg.AddBox(position=[1.5, 0.0, 0.1], euler=[0, 0, 0.0], size=[1, 1.5, 0.2])
    
    # Geometry obstacle (cylinder)
    tg.AddGeometry(position=[1.5, 0.0, 0.25], euler=[0, 0, 0.0], size=[1.0,0.5,0.5], geo_type="cylinder")

    # Slope
    tg.AddBox(position=[2.0, 2.0, 0.5],
              euler=[0.0, -0.5, 0.0],
              size=[3, 1.5, 0.1])

    # Stairs
    tg.AddStairs(init_pos=[1.0, 4.0, 0.0], yaw=0.0)

    # Rough ground
    tg.AddRoughGround(init_pos=[-2.5, 5.0, 0.0],
                      euler=[0, 0, 0.0],
                      nums=[10, 8])

    # Perlin height field
    tg.AddPerlinHeighField(position=[-1.5, 4.0, 0.0], size=[2.0, 1.5])

    tg.Save()
