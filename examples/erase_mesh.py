import pyglet.app
from pyglet.window import key
import trimesh
import numpy as np


class BoolWrapper:
    def __init__(self):
        self.val = False


class IntWrapper:
    def __init__(self):
        self.val = 0


def find_vector_direction(x: int, y: int, scene: trimesh.scene.Scene):
    """
    Find the unit vector in the direction of a pixel on the screen

    :param x: the x-coordinate of the pixel (distance from left edge)
    :param y: the y-coordinate of the pixel (distance from bottom edge)
    :param scene: the trimesh Scene that is being viewed
    :return: a 3D numpy vector giving the direction of the points represented
        by that pixel
    """
    _, drctns, pixels = scene.camera_rays()
    rows = np.where((pixels[:, 0] == x) & (pixels[:, 1] == y))
    row = rows[0][0]
    return drctns[row, :]


def main():
    src = '2011HondaOdysseyScan1.glb'
    mesh: trimesh.primitives.Trimesh = trimesh.load(src, force='mesh')
    scene = trimesh.Scene(mesh)
    res_x, res_y = 1280, 666
    fov_y = 45
    focal_len = res_y / 2.0 / np.tan(np.radians(fov_y) / 2.0)
    fov_x = 2.0 * np.degrees(np.arctan(res_x / 2.0 / focal_len))
    scene.set_camera(resolution=(1280, 666), fov=(fov_x, fov_y))
    viewer = scene.show(start_loop=False, callback=lambda s: None)
    is_e_held = BoolWrapper()
    is_d_held = BoolWrapper()

    @viewer.event
    def on_key_press(symbol, modifiers):
        if symbol == key.E:
            print('E pressed')
            is_e_held.val = True
        elif symbol == key.D:
            print('D pressed')
            is_d_held.val = True

    @viewer.event
    def on_key_release(symbol, modifiers):
        if symbol == key.E:
            print('E released')
            is_e_held.val = False
        elif symbol == key.D:
            print('D released')
            is_d_held.val = False

    @viewer.event
    def on_mouse_motion(x, y, dx, dy):
        if is_e_held.val or is_d_held.val:
            origin = viewer.scene.camera_transform[:3, 3]
            drctn = find_vector_direction(x, y, viewer.scene)
            scene_mesh: trimesh.Trimesh = viewer.scene.geometry['GLTF']
            print('Casting')
            tri_idx = scene_mesh.ray.intersects_first(origin.reshape((1, 3)), drctn.reshape((1, 3)))[0]

            if is_e_held.val:
                face_mask = np.ones(scene_mesh.faces.shape[0], dtype=bool)
                face_mask[tri_idx] = False
                scene_mesh.update_faces(face_mask)
            elif is_d_held.val:
                print('Drawing')
                endpoint = scene_mesh.triangles_center[tri_idx, :]
                ray = trimesh.load_path(np.array([origin, endpoint]))
                viewer.scene.add_geometry(ray)

    pyglet.app.run()


if __name__ == '__main__':
    main()
