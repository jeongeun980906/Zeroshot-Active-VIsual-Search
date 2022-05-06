import ai2thor
from ai2thor.controller import Controller,BFSController
from ai2thor.platform import CloudRendering
from ithor_tools.vis_tool import *
from ithor_tools.transform import cornerpoint_projection
from ithor_tools.map import single_scenemap
from ithor_tools.astar import astar
import random
import math
from IPython.display import display
from moviepy.editor import ImageSequenceClip,VideoFileClip
from matplotlib import transforms

np.random.seed(42)
random.seed(42)

"AI2-THOR Version: " + ai2thor.__version__
gridSize=0.05

controller = Controller(
    # platform = CloudRendering,
    agentMode="locobot",
    visibilityDistance=1.5,
    scene="FloorPlan_Train1_3",
    gridSize=gridSize,
    movementGaussianSigma=0,
    rotateStepDegrees=90,
    rotateGaussianSigma=0,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    width=500,
    height=500,
    fieldOfView=60
)
controller.reset(
    # makes the images a bit higher quality
    width=400,
    height=400,

    # Renders several new image modalities
    renderDepthImage=True,
    renderInstanceSegmentation=False,
    renderSemanticSegmentation=False,
    renderNormalsImage=False
)

scene_bounds = controller.last_event.metadata['sceneBounds']['center']
controller.step(
    action="AddThirdPartyCamera",
    position=dict(x=scene_bounds['x'], y=5, z=scene_bounds['z']),
    rotation=dict(x=90, y=0, z=0),
    orthographic=True,
    orthographicSize=5, # size of output image
    skyboxColor="white",
    fieldOfView = 90
)


controller.step(dict(action='GetReachablePositions'))
rstate = controller.last_event.metadata['actionReturn']

verbose = False

initstate = 20
for goalstate in range(300, 3000,100):

    controller.step(
        action="Teleport",
        position = rstate[initstate]
    )


    from RRT import gridmaprrt as rrt

    rrtplanner = rrt.RRT(controller = controller, expand_dis=0.1,max_iter=10000,goal_sample_rate=20)



    rrtplanner.set_start(rstate[initstate])
    rrtplanner.set_goal(rstate[goalstate])
    path = rrtplanner.planning(animation=False) # Uncomment "%matplotlib tk" when you want to animate
    # rrtplanner.plot_path(path)
    # flag = rrtplanner.get_Navigation_success_flag(path,verbose=verbose)

    from RRT import gridmaprrt_pathsmoothing as smoothing
    smoothpath = smoothing.path_smoothing(rrtplanner,40,verbose=verbose)

    # rrtplanner.plot_path(smoothpath)
    flag = rrtplanner.get_Navigation_success_flag(smoothpath,verbose=verbose)

    if not flag:
        print(goalstate)


pass