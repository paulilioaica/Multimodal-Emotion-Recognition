import time

import numpy as np
from vpython import *
points = ["spinebase", "spinemid", "neck", "head", "shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
          "wrist_left", "wrist_right", "hand_left", "hand_right", "hip_left", "hip_right", "knee_left", "knee_right",
          "ankle_left", "ankle_right", "foot_left", "foot_right"]
indx = {"spinebase": 2, "spinemid": 8, "neck": 14, "head": 20, "shoulder_left": 26, "shoulder_right": 50,
        "elbow_left": 32, "elbow_right": 56,
        "wrist_left": 38, "wrist_right": 62, "hand_left": 44, "hand_right": 68, "hip_left": 74, "hip_right": 98,
        "knee_left": 80, "knee_right": 104,
        "ankle_left": 86, "ankle_right": 110, "foot_left": 92, "foot_right": 116}


def read_file(str):
    fl = np.load(str)['arr_0']
    return fl


class Joint:
    def __init__(self, x ,y ,z):
        self._x = x
        self._y = y
        self._z = z

class Point:
    def __init__(self, name, x, y, z, id):
        self.name = name
        self.id = id
        self.x = x
        self.y = y
        self.z = z

    def update(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


poi = []
for i, point in enumerate(points):
    poi.append(Point(points[i], 0, 0, 0, i))


class Skeleton:
    """Kinect skeleton represented as a VPython frame.
    """

    def __init__(self, f, file):
        """Create a skeleton in the given VPython frame f.
        """
        self.frame = f
        self.file = file
        self.joints = [sphere(frame=f, radius=0.05, color=color.yellow)
                       for i in range(len(points))]
        self.joints[3].radius = 0.125
        self.bones = [cylinder(frame=f, radius=0.05, color=color.yellow)
                      for i in range(len(_bone_ids))]

    def update(self):
        """Update the skeleton joint positions in the depth sensor frame.
        Return true iff the most recent sensor frame contained a tracked
        skeleton.
        """
        updated = False
        for slice in self.file[1:]:
            time.sleep(0.01)
            # Move the joints.
            for joint, p in zip(self.joints, [Joint(float(slice[i]), float(slice[i+1]), -float(slice[i+2])) for i in indx.values()]):
                joint.pos = p

            # Move the bones.
            for bone, bone_id in zip(self.bones, _bone_ids):
                p1, p2 = [self.joints[id].pos for id in bone_id]
                bone.pos = p1
                bone.axis = p2 - p1
            updated = True
        return updated

for i in range(len(points)):
    print("{}: {}".format(points[i], i))
print(len(points))
# A bone is a cylinder connecting two joints, each specified by an id.
_bone_ids = [[0, 1], [1, 2], [2, 4], [4, 6], [6, 8], [8, 10], [2, 5], [5, 7], [7, 9], [9, 11], [0, 12], [0, 13],
             [12, 14], [14, 16], [16, 18], [13, 15], [15, 17], [17, 19]]
# Initialize and level the Kinect sensor.

if __name__ == '__main__':
    skeleton = Skeleton(f=None,file=read_file(r"C:\Users\Paul\Desktop\FG2020\dataset\motion capture\F3.6.5.npz"))
    while True:
        rate(30)
        skeleton.update()
