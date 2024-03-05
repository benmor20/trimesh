import trimesh
import numpy as np
import pyglet
from pyglet.window import key
from typing import *


def create_box(start: Sequence[float], end: Sequence[float]) -> np.array:
    """
    Create an array of line segments that form a box that spans from start to end

    :param start: a 3D numpy vector giving one corner of the box
    :param end: a 3D numpy vector giving the other corner of the box
    :return: a 12x2x3 numpy array giving the start and end points for the 12
        segments in the box, in no particular order
    """
    return np.array([[[start[0], start[1], start[2]], [end[0], start[1], start[2]]],
                     [[start[0], start[1], start[2]], [start[0], end[1], start[2]]],
                     [[start[0], start[1], start[2]], [start[0], start[1], end[2]]],
                     [[end[0], end[1], end[2]], [end[0], end[1], start[2]]],
                     [[end[0], end[1], end[2]], [end[0], start[1], end[2]]],
                     [[end[0], end[1], end[2]], [start[0], end[1], end[2]]],
                     [[start[0], start[1], end[2]], [start[0], end[1], end[2]]],
                     [[start[0], end[1], end[2]], [start[0], end[1], start[2]]],
                     [[start[0], end[1], start[2]], [end[0], end[1], start[2]]],
                     [[end[0], end[1], start[2]], [end[0], start[1], start[2]]],
                     [[end[0], start[1], start[2]], [end[0], start[1], end[2]]],
                     [[end[0], start[1], end[2]], [start[0], start[1], end[2]]]])


class BoolWrapper:
    def __init__(self):
        self.val = False


def main():
    # Display a 2x2x2 box in the middle of the scene
    box = create_box(np.array([-1, -1, -1]), np.array([1, 1, 1]))
    res_x, res_y = 1280, 666
    fov_y = 45
    focal_y = res_y / 2 / np.tan(np.radians(fov_y / 2))
    focal_x = focal_y * res_x / res_y
    fov_x = np.degrees(2 * np.arctan(res_x / 2 / focal_x))
    print(f'{fov_x = }, {fov_y = }')
    print(f'{focal_x = }, {focal_y = }')
    scene = trimesh.Scene(trimesh.load_path(box),
                          camera=trimesh.Camera(resolution=(res_x, res_y), focal=(focal_y, focal_y)),
                          camera_transform=np.eye(4))
    print(scene.camera.focal)
    print(scene.camera.fov)
    viewer = scene.show(start_loop=False, callback=lambda s: None)
    draw_active = BoolWrapper()

    @viewer.event
    def on_key_press(symbol, modifiers):
        if symbol == key.D:
            draw_active.val = True
        elif symbol == key.R:
            viewer.scene.set_camera(angles=np.zeros((3,)), center=np.zeros((3,)))
            viewer.view = {
                'cull': True,
                'axis': False,
                'grid': False,
                'fullscreen': False,
                'wireframe': False,
                'ball': trimesh.Trackball(
                    pose=np.eye(4),
                    size=viewer.scene.camera.resolution,
                    scale=viewer.scene.scale,
                    target=viewer.scene.centroid)}
        elif symbol == key.C:
            origins, drctns, pixels = viewer.scene.camera_rays()
            res = viewer.scene.camera.resolution
            idxs = np.where((pixels[:, 0] == 10)
                            | (pixels[:, 1] == 10)
                            | (pixels[:, 0] == res[0] - 10)
                            | (pixels[:, 1] == res[1] - 10))[0]
            lines = np.stack([origins[idxs, :], 5 * drctns[idxs, :]]).swapaxes(0, 1)
            rays = trimesh.load_path(lines)
            viewer.scene.add_geometry(rays)

    @viewer.event
    def on_key_release(symbol, modifiers):
        if symbol == key.D:
            draw_active.val = False

    @viewer.event
    def on_mouse_press(x, y, buttons, modifiers):
        print(f'Clicked at {x}, {y} (res is {viewer.scene.camera.resolution})')

    @viewer.event
    def on_mouse_motion(x, y, dx, dy):
        """
        When the mouse moves in the scene, send a 5m vector from the camera towards
        the direction of the mouse
        """
        if not draw_active.val:
            return
        origins, drctns, pixels = viewer.scene.camera_rays()
        idx = np.where((pixels[:, 0] == x) & pixels[:, 1] == y)[0][0]
        origin = origins[idx, :]
        drctn = drctns[idx, :]

        # Display 2 meters of the ray in the scene
        ray = np.array([origin, origin + 2 * drctn])
        viewer.scene.add_geometry(trimesh.load_path(ray))

    pyglet.app.run()


if __name__ == '__main__':
    main()
