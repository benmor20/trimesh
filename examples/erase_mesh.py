import pyglet.app
from numpy._typing import ArrayLike
from pyglet.window import key
from rtree import Index

import trimesh
import numpy as np


class BoolWrapper:
    def __init__(self):
        self.val = False


class IntWrapper:
    def __init__(self):
        self.val = 0


class ErasableTrimesh(trimesh.Trimesh):
    def __init__(self, *args, **kwargs):
        self._triangles_tree = None
        self._made_triangles_tree = False
        self._mesh_to_rtree_index = None
        self._rtree_to_mesh_index = None
        self._valid_rtree_to_mesh_indices = None
        self._reset_rtree_cache = True
        if len(args) == 1 and isinstance(args[0], trimesh.Trimesh):
            mesh: trimesh.Trimesh = args[0]
            super().__init__(
                mesh.vertices,
                mesh.faces,
                mesh.face_normals,
                mesh._cache["vertex_normals"],
            )
        else:
            super().__init__(*args, **kwargs)

    @faces.setter
    def faces(self, values):
        super().faces = values
        if self._reset_rtree_cache:
            self._made_triangles_tree = False
            if values is None or len(values) == 0:
                self._mesh_to_rtree_index = None
                self._rtree_to_mesh_index = None
                self._valid_rtree_to_mesh_indices = None
            else:
                self._mesh_to_rtree_index = np.arange(len(values))
                self._rtree_to_mesh_index = self._mesh_to_rtree_index.copy()
                self._valid_rtree_to_mesh_indices = np.ones(len(values), dtype=bool)

    @vertices.setter
    def vertices(self, values):
        super().vertices = values
        if self._reset_rtree_cache:
            self._made_triangles_tree = False

    @trimesh.caching.cache_decorator
    def triangle_bounds(self):
        """
        Bounding boxes for the triangles in this mesh
        Returns
        -------
        bounds: (n, 6) float
            interleaved bounding box for every triangle
        """
        return trimesh.triangles.triangle_bounding_boxes(self.triangles)

    @property
    def triangles_tree(self) -> Index:
        if not self._made_triangles_tree:
            print('Recalculating tree')
            self._triangles_tree = trimesh.triangles.bounds_tree(self.triangles)
            self._made_triangles_tree = True
        return self._triangles_tree

    def update_faces(self, mask: ArrayLike) -> None:
        self._reset_rtree_cache = False
        super().update_faces(mask)
        self._reset_rtree_cache = True

        inv_mask = ~mask
        idxs_to_remove = self._mesh_to_rtree_index[inv_mask]
        boxes_to_remove = self.triangle_bounds[inv_mask, :]
        for idx, box in zip(idxs_to_remove, boxes_to_remove):
            self._triangles_tree.delete(idx, box)
        self._mesh_to_rtree_index = self._mesh_to_rtree_index[mask]
        self._rtree_to_mesh_index[self._valid_rtree_to_mesh_indices] -= np.cumsum(inv_mask)
        self._valid_rtree_to_mesh_indices[self._valid_rtree_to_mesh_indices] = mask


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
    og_mesh: trimesh.Trimesh = trimesh.load(src, force='mesh')
    mesh = ErasableTrimesh(og_mesh)
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
