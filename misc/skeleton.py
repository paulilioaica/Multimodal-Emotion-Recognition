import os
import time
from PIL import ImageGrab
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
    def __init__(self, x ,y ,z, anglex, angley, anglez):
        self._x = x
        self._y = y
        self._z = z
        self._anglex= anglex
        self._angley= angley
        self._anglez= anglez


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

    def __init__(self, f, file):
        self.frame = f
        self.file = file
        self.joints = [sphere(frame=f, radius=0.05, color=color.yellow)
                       for i in range(len(points))]
        self.joints[3].radius = 0.125
        self.bones = [cylinder(frame=f, radius=0.05, color=color.yellow)
                      for i in range(len(_bone_ids))]

    def update(self, j):
        slice = self.file[j]
        for joint, p in zip(self.joints, [Joint(float(slice[i]), float(slice[i+1]), -float(slice[i+2]), float(slice[i+3]), float(slice[i+4]), float(slice[i+5])) for i in indx.values()]):
            joint.pos = p
            joint.rotate(p._anglex, axis=(vec(p._x, 0,0)))
            joint.rotate(p._angley, axis=(vec(0, p._y, 0)))
            joint.rotate(p._anglez, axis=(vec(0,0,p._z)))


        # Move the bones.
        for bone, bone_id in zip(self.bones, _bone_ids):
            p1, p2 = [self.joints[id].pos for id in bone_id]
            bone.pos = p1
            bone.axis = p2 - p1
    def delete(self):
        for joint in self.joints:
            joint.visible = False
            del joint
        for bone in self.bones:
            bone.visible = False
            del bone


for i in range(len(points)):
    print("{}: {}".format(points[i], i))
print(len(points))
# A bone is a cylinder connecting two joints, each specified by an id.
_bone_ids = [[0, 1], [1, 2], [2, 4], [4, 6], [6, 8], [8, 10], [2, 5], [5, 7], [7, 9], [9, 11], [0, 12], [0, 13],
             [12, 14], [14, 16], [16, 18], [13, 15], [15, 17], [17, 19]]
# Initialize and level the Kinect sensor.

if __name__ == '__main__':
    f = canvas(title="Exp", width=300, height=300, center=vector(0, 0, -2.5), background=color.black)
    for file in os.listdir(r"C:\Users\Paul\Desktop\FG2020\dataset\motion capture"):
        image = None
        print(file)
        fl = read_file(os.path.join(r"C:\Users\Paul\Desktop\FG2020\dataset\motion capture",file))
        skeleton = Skeleton(f=None, file=fl)
        img = np.zeros((40, 200, 200, 3))
        ii = [int(x) for x in np.linspace(2, fl.shape[0]-1, 40)]
        pos = 0
        for i in range(fl.shape[0]):
            skeleton.update(i)
            if i == 0:
                time.sleep(1)
            rate(24)
            if i in ii:
                prnt = ImageGrab.grab(bbox=(103,220,303, 420))
                x = np.array(prnt, dtype=np.uint8)
                img[pos] = x
                pos += 1
        np.savez(os.path.join(r"C:\Users\Paul\Desktop\Kinect", file), img)
        skeleton.delete()
