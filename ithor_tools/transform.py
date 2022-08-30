import numpy as np
import open3d as o3d
import math
import copy

def cornerpoint_projection(cornerpoints):
    res = []
    for e,c in enumerate(cornerpoints):
        if e%4==0 or e%4==1:
            res.append([c[0],c[2]])
    return res

def ndarray(list):
    array = np.array(list)
    return array

def to_rad(th):
    return th*math.pi / 180

class depth2world:
    def __init__(self,width=800,height=800,fov=60):
        self.width = width
        self.height = height
        self.fov = fov
        # camera intrinsics
        self.focal_length = 0.5 * width / math.tan(to_rad(fov/2))
        fx, fy, cx, cy = (self.focal_length, self.focal_length, width/2, height/2)
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, 
                                fx, fy, cx, cy)
    # Obtain point cloud
    def object_coord(self,candidate_boxes,DEPTH,COLOR,agent_pos,agent_rot):
        '''
        returns: [Dict(x,y,z)] of candidate boxes
        '''
        gray = (COLOR.sum(axis=-1)/(3*255)).astype(np.float32)
        depth = o3d.geometry.Image(DEPTH)
        color = o3d.geometry.Image(gray)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                                                                    depth_scale=1.0,
                                                                    depth_trunc=3.0,
                                                                    convert_rgb_to_intensity=False)
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        rot = agent_rot['y']
        rot = math.pi*rot/180
        pcd.transform([[-math.sin(rot), 0,-math.cos(rot), agent_pos['z']],
                [0, 1, 0, agent_pos['y']],
                [math.cos(rot), 0, -math.sin(rot), agent_pos['x']],
                [0, 0, 0, 1]])
        points = np.asarray(pcd.points)
        points = np.resize(points, (self.width,self.height,3))
        res = []
        for candidate_box in candidate_boxes:
            crop_points = copy.deepcopy(points[candidate_box[1]:candidate_box[3],candidate_box[0]:candidate_box[2],:])
            new_height = crop_points.shape[0]
            new_width = crop_points.shape[1]
            center_points = crop_points[int(2*new_height/5):int(3*new_height/5),int(2*new_width/5):int(3*new_width/5),:]
            center_points = center_points.reshape(-1,3)
            center_points = center_points.mean(axis=0)
            res.append(dict(x=center_points[2],y=center_points[1],z=center_points[0]))
        return res