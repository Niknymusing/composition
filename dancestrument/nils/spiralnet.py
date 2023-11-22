import torch
from torch.functional import F
import torch.nn as nn
from torch_scatter import scatter_add
import torch.optim as optim
import math
import heapq
import numpy as np
import os
import scipy.sparse as sp
#from mesh_utils import Mesh #from psbody.mesh import Mesh
import openmesh as om
from sklearn.neighbors import KDTree
import numpy as np

import threading
import numpy as np
import torch
import socket
from pythonosc import osc_server
from pythonosc.dispatcher import Dispatcher
from pythonosc import udp_client
import socketserver
import time


"""
Error heirarchy for Mesh class
"""


class MeshError(Exception):
    """Base error class for Mesh-related errors"""
    pass


class SerializationError(MeshError):
    """Mesh reading or writing errors"""
    pass


import numpy as np

"""
landmarks.py

"""


def landm_xyz_linear_transform(self, ordering=None):
    from .utils import col, sparse

    landmark_order = ordering if ordering else self.landm_names
    # construct a sparse matrix that converts between the landmark pts and all vertices, with height (# landmarks * 3) and width (# vertices * 3)
    if hasattr(self, 'landm_regressors'):
        landmark_coefficients = np.hstack([self.landm_regressors[name][1] for name in landmark_order])
        landmark_indices = np.hstack([self.landm_regressors[name][0] for name in landmark_order])
        column_indices = np.hstack([col(3 * landmark_indices + i) for i in range(3)]).flatten()
        row_indices = np.hstack([[3 * index, 3 * index + 1, 3 * index + 2] * len(self.landm_regressors[landmark_order[index]][0]) for index in np.arange(len(landmark_order))])
        values = np.hstack([col(landmark_coefficients) for i in range(3)]).flatten()
        return sparse(row_indices, column_indices, values, 3 * len(landmark_order), 3 * self.v.shape[0])
    elif hasattr(self, 'landm'):
        landmark_indices = np.array([self.landm[name] for name in landmark_order])
        column_indices = np.hstack(([col(3 * landmark_indices + i) for i in range(3)])).flatten()
        row_indices = np.arange(3 * len(landmark_order))
        return sparse(row_indices, column_indices, np.ones(len(column_indices)), 3 * len(landmark_order), 3 * self.v.shape[0])
    else:
        return np.zeros((0, 0))


@property
def landm_xyz(self, ordering=None):
    landmark_order = ordering if ordering else self.landm_names
    landmark_vertex_locations = (self.landm_xyz_linear_transform(landmark_order) * self.v.flatten()).reshape(-1, 3) if landmark_order else np.zeros((0, 0))
    if landmark_order:
        return dict([(landmark_order[i], xyz) for i, xyz in enumerate(landmark_vertex_locations)])
    return {}


def recompute_landmark_indices(self, landmark_fname=None, safe_mode=True):
    filtered_landmarks = dict(
        filter(
            lambda e, : e[1] != [0.0, 0.0, 0.0],
            self.landm_raw_xyz.items()
        ) if (landmark_fname and safe_mode) else self.landm_raw_xyz.items())
    if len(filtered_landmarks) != len(self.landm_raw_xyz):
        print("WARNING: %d landmarks in file %s are positioned at (0.0, 0.0, 0.0) and were ignored" % (len(self.landm_raw_xyz) - len(filtered_landmarks), landmark_fname))

    self.landm = {}
    self.landm_regressors = {}
    if filtered_landmarks:
        landmark_names = list(filtered_landmarks.keys())
        closest_vertices, _ = self.closest_vertices(np.array(list(filtered_landmarks.values())))
        self.landm = dict(zip(landmark_names, closest_vertices))
        if len(self.f):
            face_indices, closest_points = self.closest_faces_and_points(np.array(list(filtered_landmarks.values())))
            vertex_indices, coefficients = self.barycentric_coordinates_for_points(closest_points, face_indices)
            self.landm_regressors = dict([(name, (vertex_indices[i], coefficients[i])) for i, name in enumerate(landmark_names)])
        else:
            self.landm_regressors = dict([(name, (np.array([closest_vertices[i]]), np.array([1.0]))) for i, name in enumerate(landmark_names)])


def set_landmarks_from_xyz(self, landm_raw_xyz):
    self.landm_raw_xyz = landm_raw_xyz if hasattr(landm_raw_xyz, 'keys') else dict((str(i), l) for i, l in enumerate(landm_raw_xyz))
    self.recompute_landmark_indices()


def is_vertex(x):
    return hasattr(x, "__len__") and len(x) == 3


def is_index(x):
    return isinstance(x, (int, np.int32, np.int64))


def set_landmarks_from_raw(self, landmarks):
    '''
    can accept:
    {'name1': [float, float, float], 'name2': [float, float, float], ...}
    {'name1': np.array([float, float, float]), 'name2': np.array([float, float, float]), ...}
    [[float,float,float],[float,float,float], ...]
    np.array([[float,float,float],[float,float,float], ...])
    [np.array([float,float,float]),np.array([float,float,float]), ...]
    {'name1': int, 'name2': int, ...}
    [int,int,int]
    np.array([int,int,int])
    '''
    landmarks = landmarks if hasattr(landmarks, 'keys') else dict((str(i), l) for i, l in enumerate(landmarks))

    if all(is_vertex(x) for x in landmarks.values()):
        landmarks = dict((i, np.array(l)) for i, l in landmarks.items())
        self.set_landmarks_from_xyz(landmarks)
    elif all(is_index(x) for x in landmarks.values()):
        self.landm = landmarks
        self.recompute_landmark_xyz()
    else:
        raise Exception("Can't parse landmarks")
    


"""
Mesh processing backend
=======================

"""

import numpy as np


def reset_normals(self, face_to_verts_sparse_matrix=None, reset_face_normals=False):
    self.vn = self.estimate_vertex_normals(face_to_verts_sparse_matrix=None)
    if reset_face_normals:
        self.fn = self.f.copy()
    return self


def reset_face_normals(self):
    if not hasattr(self, 'vn'):
        self.reset_normals()
    self.fn = self.f
    return self


def uniquified_mesh(self):
    """This function returns a copy of the mesh in which vertices are copied such that
    each vertex appears in only one face, and hence has only one texture"""
    from mesh import Mesh
    new_mesh = Mesh(v=self.v[self.f.flatten()], f=np.array(range(len(self.f.flatten()))).reshape(-1, 3))

    if not hasattr(self, 'vn'):
        self.reset_normals()
    new_mesh.vn = self.vn[self.f.flatten()]

    if hasattr(self, 'vt'):
        new_mesh.vt = self.vt[self.ft.flatten()]
        new_mesh.ft = new_mesh.f.copy()
    return new_mesh


def keep_vertices(self, keep_list):
    trans = dict((v, i) for i, v in enumerate(keep_list))
    trans_f = np.array([trans[v] if v in trans else -1 for row in self.f for v in row], dtype=np.uint32).reshape(-1, 3)
    if hasattr(self, 'vn') and self.vn.shape[0] == self.vn.shape[0]:
        self.vn = self.vn.reshape(-1, 3)[keep_list]
    if hasattr(self, 'vc') and self.vc.shape[0] == self.v.shape[0]:
        self.vc = self.vc.reshape(-1, 3)[keep_list]
    if hasattr(self, 'landm_raw_xyz'):
        self.recompute_landmark_indices()

    self.v = self.v.reshape(-1, 3)[keep_list]
    self.f = trans_f[(trans_f != np.uint32(-1)).all(axis=1)]
    return self


def point_cloud(self):
    from .mesh import Mesh
    return Mesh(v=self.v, f=[], vc=self.vc) if hasattr(self, 'vc') else Mesh(v=self.v, f=[])


def remove_faces(self, face_indices_to_remove):

    def arr_replace(arr_in, lookup_dict):
        arr_out = arr_in.copy()
        for k, v in lookup_dict.iteritems():
            arr_out[arr_in == k] = v
        return arr_out

    f = np.delete(self.f, face_indices_to_remove, 0)
    v2keep = np.unique(f)
    self.v = self.v[v2keep]
    self.f = arr_replace(f, dict((v, i) for i, v in enumerate(v2keep)))

    if hasattr(self, 'fc'):
        self.fc = np.delete(self.fc, face_indices_to_remove, 0)
    if hasattr(self, 'vn') and self.vn.shape[0] == self.vn.shape[0]:
        self.vn = self.vn.reshape(-1, 3)[v2keep]
    if hasattr(self, 'vc') and self.vc.shape[0] == self.v.shape[0]:
        self.vc = self.vc.reshape(-1, 3)[v2keep]
    if hasattr(self, 'landm_raw_xyz'):
        self.recompute_landmark_indices()

    if hasattr(self, 'ft'):
        ft = np.delete(self.ft, face_indices_to_remove, 0)
        vt2keep = np.unique(ft)
        self.vt = self.vt[vt2keep]
        self.ft = arr_replace(ft, dict((v, i) for i, v in enumerate(vt2keep)))

    return self


def flip_faces(self):
    self.f = self.f.copy()
    for i in range(len(self.f)):
        self.f[i] = self.f[i][::-1]
    if hasattr(self, 'ft'):
        for i in range(len(self.f)):
            self.ft[i] = self.ft[i][::-1]
    return self


def scale_vertices(self, scale_factor):
    self.v *= scale_factor
    return self


def rotate_vertices(self, rotation_matrix):
    import cv2
    rotation_matrix = np.matrix(cv2.Rodrigues(np.array(rotation_matrix))[0] if (np.array(rotation_matrix).shape != (3, 3)) else rotation_matrix)
    self.v = np.array(self.v * rotation_matrix.T)
    return self


def translate_vertices(self, translation):
    self.v += translation
    return self


def subdivide_triangles(self):
    new_faces = []
    new_vertices = self.v.copy()
    for face in self.f:
        face_vertices = np.array([self.v[face[0], :], self.v[face[1], :], self.v[face[2], :]])
        new_vertex = np.mean(face_vertices, axis=0)
        new_vertices = np.vstack([new_vertices, new_vertex])
        new_vertex_index = len(new_vertices) - 1
        if len(new_faces):
            new_faces = np.vstack([new_faces, [face[0], face[1], new_vertex_index], [face[1], face[2], new_vertex_index], [face[2], face[0], new_vertex_index]])
        else:
            new_faces = np.array([[face[0], face[1], new_vertex_index], [face[1], face[2], new_vertex_index], [face[2], face[0], new_vertex_index]])
    self.v = new_vertices
    self.f = new_faces

    if hasattr(self, 'vt'):
        new_ft = []
        new_texture_coordinates = self.vt.copy()
        for face_texture in self.ft:
            face_texture_coordinates = np.array([self.vt[face_texture[0], :], self.vt[face_texture[1], :], self.vt[face_texture[2], :]])
            new_texture_coordinate = np.mean(face_texture_coordinates, axis=0)
            new_texture_coordinates = np.vstack([new_texture_coordinates, new_texture_coordinate])
            new_texture_index = len(new_texture_coordinates) - 1
            if len(new_ft):
                new_ft = np.vstack([new_ft, [face_texture[0], face_texture[1], new_texture_index], [face_texture[1], face_texture[2], new_texture_index], [face_texture[2], face_texture[0], new_texture_index]])
            else:
                new_ft = np.array([[face_texture[0], face_texture[1], new_texture_index], [face_texture[1], face_texture[2], new_texture_index], [face_texture[2], face_texture[0], new_texture_index]])
        self.vt = new_texture_coordinates
        self.ft = new_ft
    return self


def concatenate_mesh(self, mesh):
    if len(self.v) == 0:
        self.f = mesh.f.copy()
        self.v = mesh.v.copy()
        self.vc = mesh.vc.copy() if hasattr(mesh, 'vc') else None
    elif len(mesh.v):
        self.f = np.concatenate([self.f, mesh.f.copy() + len(self.v)])
        self.v = np.concatenate([self.v, mesh.v])
        self.vc = np.concatenate([self.vc, mesh.vc]) if (hasattr(mesh, 'vc') and hasattr(self, 'vc')) else None
    return self


# new_ordering specifies the new index of each vertex. If new_ordering[i] = j,
# vertex i should now be the j^th vertex. As such, each entry in new_ordering should be unique.
def reorder_vertices(self, new_ordering, new_normal_ordering=None):
    if new_normal_ordering is None:
        new_normal_ordering = new_ordering
    inverse_ordering = np.zeros(len(new_ordering), dtype=int)
    for i, j in enumerate(new_ordering):
        inverse_ordering[j] = i
    inverse_normal_ordering = np.zeros(len(new_normal_ordering), dtype=int)
    for i, j in enumerate(new_normal_ordering):
        inverse_normal_ordering[j] = i
    self.v = self.v[inverse_ordering]
    if hasattr(self, 'vn'):
        self.vn = self.vn[inverse_normal_ordering]
    for i in range(len(self.f)):
        self.f[i] = np.array([new_ordering[vertex_index] for vertex_index in self.f[i]])
        if hasattr(self, 'fn'):
            self.fn[i] = np.array([new_normal_ordering[normal_index] for normal_index in self.fn[i]])




"""
Searching and lookup of geometric entities
==========================================

"""


import numpy as np

__all__ = ['AabbTree', 'AabbNormalsTree', 'ClosestPointTree', 'CGALClosestPointTree']


class AabbTree(object):
    """Encapsulates an AABB (Axis Aligned Bounding Box) Tree"""
    def __init__(self, m):
        from . import spatialsearch
        # this shit return NULL
        self.cpp_handle = spatialsearch.aabbtree_compute(m.v.astype(np.float64).copy(order='C'), m.f.astype(np.uint32).copy(order='C'))

    def nearest(self, v_samples, nearest_part=False):
        "nearest_part tells you whether the closest point in triangle abc is in the interior (0), on an edge (ab:1,bc:2,ca:3), or a vertex (a:4,b:5,c:6)"
        from . import spatialsearch
        f_idxs, f_part, v = spatialsearch.aabbtree_nearest(self.cpp_handle, np.array(v_samples, dtype=np.float64, order='C'))
        return (f_idxs, f_part, v) if nearest_part else (f_idxs, v)

    def nearest_alongnormal(self, points, normals):
        from . import spatialsearch
        distances, f_idxs, v = spatialsearch.aabbtree_nearest_alongnormal(self.cpp_handle,
                                                                          points.astype(np.float64),
                                                                          normals.astype(np.float64))
        return (distances, f_idxs, v)

    def intersections_indices(self, q_v, q_f):
        '''
            Given a set of query vertices and faces, the function computes which intersect the mesh
            A list with the indices in q_f is returned
            @param q_v The query vertices (array of 3xN float values)
            @param q_f The query faces (array 3xF integer values)
        '''
        import spatialsearch
        return spatialsearch.aabbtree_intersections_indices(self.cpp_handle,
                                                            q_v.astype(np.float64),
                                                            q_f.astype(np.uint32))


class ClosestPointTree(object):
    """Provides nearest neighbor search for a cloud of vertices (i.e. triangles are not used)"""
    def __init__(self, m):
        from scipy.spatial import KDTree
        self.v = m.v
        self.kdtree = KDTree(self.v)

    def nearest(self, v_samples):
        (distances, indices) = zip(*[self.kdtree.query(v) for v in v_samples])
        return (indices, distances)

    def nearest_vertices(self, v_samples):
        (distances, indices) = zip(*[self.kdtree.query(v) for v in v_samples])
        return self.v[indices]


class CGALClosestPointTree(object):
    """Encapsulates an AABB (Axis Aligned Bounding Box) Tree """
    def __init__(self, m):
        from . import spatialsearch
        self.v = m.v
        n = m.v.shape[0]
        faces = np.vstack([np.array(range(n)), np.array(range(n)) + n, np.array(range(n)) + 2 * n]).T
        eps = 0.000000000001
        self.cpp_handle = spatialsearch.aabbtree_compute(np.vstack([m.v + eps * np.array([1.0, 0.0, 0.0]), m.v + eps * np.array([0.0, 1.0, 0.0]), m.v - eps * np.array([1.0, 1.0, 0.0])]).astype(np.float64).copy(order='C'), faces.astype(np.uint32).copy(order='C'))

    def nearest(self, v_samples):
        from . import spatialsearch
        f_idxs, f_part, v = spatialsearch.aabbtree_nearest(self.cpp_handle, np.array(v_samples, dtype=np.float64, order='C'))
        return (f_idxs.flatten(), (np.sum(((self.v[f_idxs.flatten()] - v_samples) ** 2.0), axis=1) ** 0.5).flatten())

    def nearest_vertices(self, v_samples):
        from . import spatialsearch
        f_idxs, f_part, v = spatialsearch.aabbtree_nearest(self.cpp_handle, np.array(v_samples, dtype=np.float64, order='C'))
        return self.v[f_idxs.flatten()]


class AabbNormalsTree(object):
    def __init__(self, m):
        # the weight of the normals cosine is proportional to the std of the vertices
        # the best point can be translated up to 2*eps because of the normals
        from . import aabb_normals
        eps = 0.1  # np.std(m.v)#0
        self.tree_handle = aabb_normals.aabbtree_n_compute(m.v, m.f.astype(np.uint32).copy(), eps)

    def nearest(self, v_samples, n_samples):
        from . import aabb_normals
        closest_tri, closest_p = aabb_normals.aabbtree_n_nearest(self.tree_handle, v_samples, n_samples)
        return (closest_tri, closest_p)
    


import numpy as np

"""
texture.py

"""

__all__ = ['texture_coordinates_by_vertex', ]


def texture_coordinates_by_vertex(self):
    texture_coordinates_by_vertex = [[] for i in range(len(self.v))]
    for i, face in enumerate(self.f):
        for j in [0, 1, 2]:
            texture_coordinates_by_vertex[face[j]].append(self.vt[self.ft[i][j]])
    return texture_coordinates_by_vertex


def reload_texture_image(self):
    import cv2
    # image is loaded as image_height-by-image_width-by-3 array in BGR color order.
    self._texture_image = cv2.imread(self.texture_filepath) if self.texture_filepath else None
    texture_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    if self._texture_image is not None and (self._texture_image.shape[0] != self._texture_image.shape[1] or
       self._texture_image.shape[0] not in texture_sizes or
       self._texture_image.shape[0] not in texture_sizes):
        closest_texture_size_idx = (np.abs(np.array(texture_sizes) - max(self._texture_image.shape))).argmin()
        sz = texture_sizes[closest_texture_size_idx]
        self._texture_image = cv2.resize(self._texture_image, (sz, sz))


def load_texture(self, texture_version):
    '''
    Expect a texture version number as an integer, load the texture version from 'texture_path' (global variable to the
    package).
    Currently there are versions [0,1,2,3] available.
    '''
    import os
    from . import texture_path

    lowres_tex_template = os.path.join(texture_path, 'textured_template_low_v%d.obj' % texture_version)
    highres_tex_template = os.path.join(texture_path, 'textured_template_high_v%d.obj' % texture_version)
    from .mesh import Mesh

    mesh_with_texture = Mesh(filename=lowres_tex_template)
    if not np.all(mesh_with_texture.f.shape == self.f.shape):
        mesh_with_texture = Mesh(filename=highres_tex_template)
    self.transfer_texture(mesh_with_texture)


def transfer_texture(self, mesh_with_texture):
    if not np.all(mesh_with_texture.f.shape == self.f.shape):
        raise Exception('Mesh topology mismatch')

    self.vt = mesh_with_texture.vt.copy()
    self.ft = mesh_with_texture.ft.copy()

    if not np.all(mesh_with_texture.f == self.f):
        if np.all(mesh_with_texture.f == np.fliplr(self.f)):
            self.ft = np.fliplr(self.ft)
        else:
            # Same shape; let's see if it's face ordering; this could be a bit faster...
            face_mapping = {}
            for f, ii in zip(self.f, range(len(self.f))):
                face_mapping[" ".join([str(x) for x in sorted(f)])] = ii
            self.ft = np.zeros(self.f.shape, dtype=np.uint32)

            for f, ft in zip(mesh_with_texture.f, mesh_with_texture.ft):
                k = " ".join([str(x) for x in sorted(f)])
                if k not in face_mapping:
                    raise Exception('Mesh topology mismatch')
                # the vertex order can be arbitrary...
                ids = []
                for f_id in f:
                    ids.append(np.where(self.f[face_mapping[k]] == f_id)[0][0])
                ids = np.array(ids)
                self.ft[face_mapping[k]] = np.array(ft[ids])

    self.texture_filepath = mesh_with_texture.texture_filepath
    self._texture_image = None


def set_texture_image(self, path_to_texture):
    self.texture_filepath = path_to_texture


def texture_rgb(self, texture_coordinate):
    h, w = np.array(self.texture_image.shape[:2]) - 1
    return np.double(self.texture_image[int(h * (1.0 - texture_coordinate[1]))][int(w * (texture_coordinate[0]))])[::-1]


def texture_rgb_vec(self, texture_coordinates):
    h, w = np.array(self.texture_image.shape[:2]) - 1
    n_ch = self.texture_image.shape[2]
    # XXX texture_coordinates can be lower than 0! clip needed!
    d1 = (h * (1.0 - np.clip(texture_coordinates[:, 1], 0, 1))).astype(np.int)
    d0 = (w * (np.clip(texture_coordinates[:, 0], 0, 1))).astype(np.int)
    flat_texture = self.texture_image.flatten()
    indices = np.hstack([((d1 * (w + 1) * n_ch) + (d0 * n_ch) + (2 - i)).reshape(-1, 1) for i in range(n_ch)])
    return flat_texture[indices]





"""
Mesh module
-----------

"""


import os
from functools import reduce

import numpy as np

#from . import colors
#from . import search

#try:
#    from .serialization import serialization
#except ImportError:
#    pass

#from . import landmarks
#from . import texture
#from . import processing


__all__ = ["Mesh"]


class Mesh(object):
    """3d Triangulated Mesh class

    Attributes:
        v: Vx3 array of vertices
        f: Fx3 array of faces

    Optional attributes:
        fc: Fx3 array of face colors
        vc: Vx3 array of vertex colors
        vn: Vx3 array of vertex normals
        segm: dictionary of part names to triangle indices

    """
    def __init__(self,
                 v=None,
                 f=None,
                 segm=None,
                 filename=None,
                 ppfilename=None,
                 lmrkfilename=None,
                 basename=None,
                 vc=None,
                 fc=None,
                 vscale=None,
                 landmarks=None):
        """
        :param v: vertices
        :param f: faces
        :param filename: a filename from which a mesh is loaded
        """

        if filename is not None:
            self.load_from_file(filename)
            if hasattr(self, 'f'):
                self.f = np.require(self.f, dtype=np.uint32)
            self.v = np.require(self.v, dtype=np.float64)
            self.filename = filename
            if vscale is not None:
                self.v *= vscale
        if v is not None:
            self.v = np.array(v, dtype=np.float64)
            if vscale is not None:
                self.v *= vscale
        if f is not None:
            self.f = np.require(f, dtype=np.uint32)

        self.basename = basename
        if self.basename is None and filename is not None:
            self.basename = os.path.splitext(os.path.basename(filename))[0]

        if segm is not None:
            self.segm = segm
        if landmarks is not None:
            self.set_landmark_indices_from_any(landmarks)
        if ppfilename is not None:
            self.set_landmark_indices_from_ppfile(ppfilename)
        if lmrkfilename is not None:
            self.set_landmark_indices_from_lmrkfile(lmrkfilename)

        if vc is not None:
            self.set_vertex_colors(vc)

        if fc is not None:
            self.set_face_colors(fc)

    def __del__(self):
        if hasattr(self, 'textureID'):
            from OpenGL.GL import glDeleteTextures
            glDeleteTextures([self.textureID])

    def edges_as_lines(self, copy_vertices=False):
        from .lines import Lines
        edges = self.f[:, [0, 1, 1, 2, 2, 0]].flatten().reshape(-1, 2)
        verts = self.v.copy() if copy_vertices else self.v
        return Lines(v=verts, e=edges)

    def show(self, mv=None, meshes=[], lines=[]):
        from .meshviewer import MeshViewer
        from .utils import row

        if mv is None:
            mv = MeshViewer(keepalive=True)

        if hasattr(self, 'landm'):
            from .sphere import Sphere
            sphere = Sphere(np.zeros((3)), 1.).to_mesh()
            scalefactor = 1e-2 * np.max(np.max(self.v) - np.min(self.v)) / np.max(np.max(sphere.v) - np.min(sphere.v))
            sphere.v = sphere.v * scalefactor
            spheres = [Mesh(vc='SteelBlue', f=sphere.f, v=sphere.v + row(np.array(self.landm_raw_xyz[k]))) for k in self.landm.keys()]
            mv.set_dynamic_meshes([self] + spheres + meshes, blocking=True)
        else:
            mv.set_dynamic_meshes([self] + meshes, blocking=True)
        mv.set_dynamic_lines(lines)
        return mv

    def colors_like(self, color, arr=None):
        from .utils import row, col

        if arr is None:
            arr = np.zeros(self.v.shape)

        # if arr is single-dim, reshape it
        if arr.ndim == 1 or arr.shape[1] == 1:
            arr = arr.reshape(-1, 3)

        if isinstance(color, str):
            q = 0 #color = colors.name_to_rgb[color]
        elif isinstance(color, list):
            color = np.array(color)

        if color.shape[0] == arr.shape[0] and color.shape[0] == color.size:
            def jet(v):
                fourValue = 4 * v
                red = min(fourValue - 1.5, -fourValue + 4.5)
                green = min(fourValue - 0.5, -fourValue + 3.5)
                blue = min(fourValue + 0.5, -fourValue + 2.5)
                result = np.array([red, green, blue])
                result[result > 1.0] = 1.0
                result[result < 0.0] = 0.0
                return row(result)
            color = col(color)
            color = np.concatenate([jet(color[i]) for i in range(color.size)], axis=0)

        return np.ones_like(arr) * color

    def set_vertex_colors(self, vc, vertex_indices=None):
        if vertex_indices is not None:
            self.vc[vertex_indices] = self.colors_like(vc, self.v[vertex_indices])
        else:
            self.vc = self.colors_like(vc, self.v)
        return self

    def set_vertex_colors_from_weights(self, weights, scale_to_range_1=True, color=True):
        # from numpy import ones_like
        if weights is None:
            return self
        if scale_to_range_1:
            weights = weights - np.min(weights)
            weights = (1.0 - 0.0) * weights / np.max(weights) + 0.0
        if color:
            from matplotlib import cm
            self.vc = cm.jet(weights)[:, :3]
        else:
            self.vc = np.tile(np.reshape(weights, (len(weights), 1)), (1, 3))  # *ones_like(self.v)
        return self

    def scale_vertex_colors(self, weights, w_min=0.0, w_max=1.0):
        if weights is None:
            return self
        weights = weights - np.min(weights)
        weights = (w_max - w_min) * weights / np.max(weights) + w_min
        self.vc = (weights * self.vc.T).T if weights is not None else self.vc
        return self

    def set_face_colors(self, fc):
        self.fc = self.colors_like(fc, self.f)
        return self

    def faces_by_vertex(self, as_sparse_matrix=False):
        import scipy.sparse as sp
        if not as_sparse_matrix:
            faces_by_vertex = [[] for i in range(len(self.v))]
            for i, face in enumerate(self.f):
                faces_by_vertex[face[0]].append(i)
                faces_by_vertex[face[1]].append(i)
                faces_by_vertex[face[2]].append(i)
        else:
            row = self.f.flatten()
            col = np.array([range(self.f.shape[0])] * 3).T.flatten()
            data = np.ones(len(col))
            faces_by_vertex = sp.csr_matrix((data, (row, col)), shape=(self.v.shape[0], self.f.shape[0]))
        return faces_by_vertex

    def estimate_vertex_normals(self, face_to_verts_sparse_matrix=None):
        from .geometry.tri_normals import TriNormalsScaled

        face_normals = TriNormalsScaled(self.v, self.f).reshape(-1, 3)
        ftov = face_to_verts_sparse_matrix if face_to_verts_sparse_matrix else self.faces_by_vertex(as_sparse_matrix=True)
        non_scaled_normals = ftov * face_normals
        norms = (np.sum(non_scaled_normals ** 2.0, axis=1) ** 0.5).T
        norms[norms == 0] = 1.0
        return (non_scaled_normals.T / norms).T

    def barycentric_coordinates_for_points(self, points, face_indices):
        from .geometry.barycentric_coordinates_of_projection import barycentric_coordinates_of_projection
        vertex_indices = self.f[face_indices.flatten(), :]
        tri_vertices = np.array([self.v[vertex_indices[:, 0]], self.v[vertex_indices[:, 1]], self.v[vertex_indices[:, 2]]])
        return vertex_indices, barycentric_coordinates_of_projection(points, tri_vertices[0, :], tri_vertices[1, :] - tri_vertices[0, :], tri_vertices[2, :] - tri_vertices[0, :])

    def transfer_segm(self, mesh, exclude_empty_parts=True):
        self.segm = {}
        if hasattr(mesh, 'segm'):
            face_centers = np.array([self.v[face, :].mean(axis=0) for face in self.f])
            (closest_faces, closest_points) = mesh.closest_faces_and_points(face_centers)
            mesh_parts_by_face = mesh.parts_by_face()
            parts_by_face = [mesh_parts_by_face[face] for face in closest_faces.flatten()]
            self.segm = dict([(part, []) for part in mesh.segm.keys()])
            for face, part in enumerate(parts_by_face):
                self.segm[part].append(face)
            for part in self.segm.keys():
                self.segm[part].sort()
                if exclude_empty_parts and not self.segm[part]:
                    del self.segm[part]

    @property
    def verts_by_segm(self):
        return dict((segment, sorted(set(self.f[indices].flatten()))) for segment, indices in self.segm.items())

    def parts_by_face(self):
        segments_by_face = [''] * len(self.f)
        for part in self.segm.keys():
            for face in self.segm[part]:
                segments_by_face[face] = part
        return segments_by_face

    def verts_in_common(self, segments):
        """
        returns array of all vertex indices common to each segment in segments"""
        return sorted(reduce(lambda s0, s1: s0.intersection(s1),
                             [set(self.verts_by_segm[segm]) for segm in segments]))
        # # indices of vertices in the faces of the first segment
        # indices = self.verts_by_segm[segments[0]]
        # for segment in segments[1:] :
        #    indices = sorted([index for index in self.verts_by_segm[segment] if index in indices]) # Intersect current segment with current indices
        # return sorted(set(indices))

    @property
    def joint_names(self):
        return self.joint_regressors.keys()

    @property
    def joint_xyz(self):
        joint_locations = {}
        for name in self.joint_names:
            joint_locations[name] = self.joint_regressors[name]['offset'] + \
                np.sum(self.v[self.joint_regressors[name]['v_indices']].T * self.joint_regressors[name]['coeff'], axis=1)
        return joint_locations

    # creates joint_regressors from a list of joint names and a per joint list of vertex indices (e.g. a ring of vertices)
    # For the regression coefficients, all vertices for a given joint are given equal weight
    def set_joints(self, joint_names, vertex_indices):
        self.joint_regressors = {}
        for name, indices in zip(joint_names, vertex_indices):
            self.joint_regressors[name] = {'v_indices': indices,
                                           'coeff': [1.0 / len(indices)] * len(indices),
                                           'offset': np.array([0., 0., 0.])}

    def vertex_visibility(self, camera, normal_threshold=None, omni_directional_camera=False, binary_visiblity=True):

        vis, n_dot_cam = self.vertex_visibility_and_normals(camera, omni_directional_camera)

        if normal_threshold is not None:
            vis = np.logical_and(vis, n_dot_cam > normal_threshold)

        return np.squeeze(vis) if binary_visiblity else np.squeeze(vis * n_dot_cam)

    def vertex_visibility_and_normals(self, camera, omni_directional_camera=False):
        from .visibility import visibility_compute
        arguments = {'v': self.v,
                     'f': self.f,
                     'cams': np.array([camera.origin.flatten()])}

        if not omni_directional_camera:
            arguments['sensors'] = np.array([camera.sensor_axis.flatten()])

        arguments['n'] = self.vn if hasattr(self, 'vn') else self.estimate_vertex_normals()

        return(visibility_compute(**arguments))

    def visibile_mesh(self, camera=[0.0, 0.0, 0.0]):
        vis = self.vertex_visibility(camera)
        faces_to_keep = filter(lambda face: vis[face[0]] * vis[face[1]] * vis[face[2]], self.f)
        vertex_indices_to_keep = np.nonzero(vis)[0]
        vertices_to_keep = self.v[vertex_indices_to_keep]
        old_to_new_indices = np.zeros(len(vis))
        old_to_new_indices[vertex_indices_to_keep] = range(len(vertex_indices_to_keep))
        return Mesh(v=vertices_to_keep, f=np.array([old_to_new_indices[face] for face in faces_to_keep]))

    def estimate_circumference(self, plane_normal, plane_distance, partNamesAllowed=None, want_edges=False):
        raise Exception('estimate_circumference function has moved to body.mesh.metrics.circumferences')

    # ######################################################
    # Processing
    def reset_normals(self, face_to_verts_sparse_matrix=None, reset_face_normals=False):
        return processing.reset_normals(self, face_to_verts_sparse_matrix, reset_face_normals)

    def reset_face_normals(self):
        return processing.reset_face_normals(self)

    def uniquified_mesh(self):
        """This function returns a copy of the mesh in which vertices are copied such that
        each vertex appears in only one face, and hence has only one texture"""
        return processing.uniquified_mesh(self)

    def keep_vertices(self, keep_list):
        return processing.keep_vertices(self, keep_list)

    def remove_vertices(self, v_list):
        return self.keep_vertices(np.setdiff1d(np.arange(self.v.shape[0]), v_list))

    def point_cloud(self):
        return Mesh(v=self.v, f=[], vc=self.vc) if hasattr(self, 'vc') else Mesh(v=self.v, f=[])

    def remove_faces(self, face_indices_to_remove):
        return processing.remove_faces(self, face_indices_to_remove)

    def scale_vertices(self, scale_factor):
        return processing.scale_vertices(self, scale_factor)

    def rotate_vertices(self, rotation):
        return processing.rotate_vertices(self, rotation)

    def translate_vertices(self, translation):
        return processing.translate_vertices(self, translation)

    def flip_faces(self):
        return processing.flip_faces(self)

    def simplified(self, factor=None, n_verts_desired=None):
        from .topology import qslim_decimator
        return qslim_decimator(self, factor, n_verts_desired)

    def subdivide_triangles(self):
        return processing.subdivide_triangles(self)

    def concatenate_mesh(self, mesh):
        return processing.concatenate_mesh(self, mesh)

    # new_ordering specifies the new index of each vertex. If new_ordering[i] = j,
    # vertex i should now be the j^th vertex. As such, each entry in new_ordering should be unique.
    def reorder_vertices(self, new_ordering, new_normal_ordering=None):
        processing.reorder_vertices(self, new_ordering, new_normal_ordering)

    # ######################################################
    # Landmark methods

    @property
    def landm_names(self):
        names = []
        if hasattr(self, 'landm_regressors') or hasattr(self, 'landm'):
            names = self.landm_regressors.keys() if hasattr(self, 'landm_regressors') else self.landm.keys()
        return list(names)

    @property
    def landm_xyz(self, ordering=None):
        landmark_order = ordering if ordering else self.landm_names
        landmark_vertex_locations = (self.landm_xyz_linear_transform(landmark_order) * self.v.flatten()).reshape(-1, 3) if landmark_order else np.zeros((0, 0))
        return dict([(landmark_order[i], xyz) for i, xyz in enumerate(landmark_vertex_locations)]) if landmark_order else {}

    def set_landmarks_from_xyz(self, landm_raw_xyz):
        self.landm_raw_xyz = landm_raw_xyz if hasattr(landm_raw_xyz, 'keys') else dict((str(i), l) for i, l in enumerate(landm_raw_xyz))
        self.recompute_landmark_indices()

    def landm_xyz_linear_transform(self, ordering=None):
        return landmarks.landm_xyz_linear_transform(self, ordering)

    def recompute_landmark_xyz(self):
        self.landm_raw_xyz = dict((name, self.v[ind]) for name, ind in self.landm.items())

    def recompute_landmark_indices(self, landmark_fname=None, safe_mode=True):
        landmarks.recompute_landmark_indices(self, landmark_fname, safe_mode)

    def set_landmarks_from_regressors(self, regressors):
        self.landm_regressors = regressors

    def set_landmark_indices_from_any(self, landmark_file_or_values):
        serialization.set_landmark_indices_from_any(self, landmark_file_or_values)

    def set_landmarks_from_raw(self, landmark_file_or_values):
        landmarks.set_landmarks_from_raw(self, landmark_file_or_values)

    #######################################################
    # Texture methods

    @property
    def texture_image(self):
        if not hasattr(self, '_texture_image'):
            self.reload_texture_image()
        return self._texture_image

    def set_texture_image(self, path_to_texture):
        self.texture_filepath = path_to_texture

    def texture_coordinates_by_vertex(self):
        return texture.texture_coordinates_by_vertex(self)

    def reload_texture_image(self):
        texture.reload_texture_image(self)

    def transfer_texture(self, mesh_with_texture):
        texture.transfer_texture(self, mesh_with_texture)

    def load_texture(self, texture_version):
        texture.load_texture(self, texture_version)

    def texture_rgb(self, texture_coordinate):
        return texture.texture_rgb(self, texture_coordinate)

    def texture_rgb_vec(self, texture_coordinates):
        return texture.texture_rgb_vec(self, texture_coordinates)

    #######################################################
    # Search methods

    def compute_aabb_tree(self):
        return search.AabbTree(self)

    def compute_aabb_normals_tree(self):
        return search.AabbNormalsTree(self)

    def compute_closest_point_tree(self, use_cgal=False):
        return search.CGALClosestPointTree(self) if use_cgal else search.ClosestPointTree(self)

    def closest_vertices(self, vertices, use_cgal=False):
        return self.compute_closest_point_tree(use_cgal).nearest(vertices)

    def closest_points(self, vertices):
        return self.closest_faces_and_points(vertices)[1]

    def closest_faces_and_points(self, vertices):
        return self.compute_aabb_tree().nearest(vertices)

    #######################################################
    # Serialization methods

    def load_from_file(self, filename):
        serialization.load_from_file(self, filename)

    def load_from_ply(self, filename):
        serialization.load_from_ply(self, filename)

    def load_from_obj(self, filename):
        serialization.load_from_obj(self, filename)

    def write_json(self, filename, header="", footer="", name="", include_faces=True, texture_mode=True):
        serialization.write_json(self, filename, header, footer, name, include_faces, texture_mode)

    def write_three_json(self, filename, name=""):
        serialization.write_three_json(self, filename, name)

    def write_ply(self, filename, flip_faces=False, ascii=False, little_endian=True, comments=[]):
        serialization.write_ply(self, filename, flip_faces, ascii, little_endian, comments)

    def write_mtl(self, path, material_name, texture_name):
        """Serializes a material attributes file"""
        serialization.write_mtl(self, path, material_name, texture_name)

    def write_obj(self, filename, flip_faces=False, group=False, comments=None):
        serialization.write_obj(self, filename, flip_faces, group, comments)

    def load_from_obj_cpp(self, filename):
        serialization.load_from_obj_cpp(self, filename)

    def set_landmark_indices_from_ppfile(self, ppfilename):
        serialization.set_landmark_indices_from_ppfile(self, ppfilename)

    def set_landmark_indices_from_lmrkfile(self, lmrkfilename):
        serialization.set_landmark_indices_from_lmrkfile(self, lmrkfilename)

# define utils for mesh processing 
def _next_ring(mesh, last_ring, other):
    res = []

    def is_new_vertex(idx):
        return (idx not in last_ring and idx not in other and idx not in res)

    for vh1 in last_ring:
        vh1 = om.VertexHandle(vh1)
        after_last_ring = False
        for vh2 in mesh.vv(vh1):
            if after_last_ring:
                if is_new_vertex(vh2.idx()):
                    res.append(vh2.idx())
            if vh2.idx() in last_ring:
                after_last_ring = True
        for vh2 in mesh.vv(vh1):
            if vh2.idx() in last_ring:
                break
            if is_new_vertex(vh2.idx()):
                res.append(vh2.idx())
    return res


def extract_spirals(mesh, seq_length, dilation=1):
    # output: spirals.size() = [N, seq_length]
    spirals = []
    for vh0 in mesh.vertices():
        reference_one_ring = []
        for vh1 in mesh.vv(vh0):
            reference_one_ring.append(vh1.idx())
        spiral = [vh0.idx()]
        one_ring = list(reference_one_ring)
        last_ring = one_ring
        next_ring = _next_ring(mesh, last_ring, spiral)
        spiral.extend(last_ring)
        while len(spiral) + len(next_ring) < seq_length * dilation:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            next_ring = _next_ring(mesh, last_ring, spiral)
            spiral.extend(last_ring)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
        else:
            kdt = KDTree(mesh.points(), metric='euclidean')
            spiral = kdt.query(np.expand_dims(mesh.points()[spiral[0]],
                                              axis=0),
                               k=seq_length * dilation,
                               return_distance=False).tolist()
            spiral = [item for subspiral in spiral for item in subspiral]
        spirals.append(spiral[:seq_length * dilation][::dilation])
    return spirals


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    #from .generate_spiral_seq import extract_spirals
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = torch.tensor(
        extract_spirals(mesh, seq_length=seq_length, dilation=dilation))
    return spirals


def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row,
                          spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data), torch.Size(spmat.tocoo().shape))


def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))


def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v), len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:, i]
        JS = mesh_f[:, (i + 1) % 3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.ravel()), row(JS.ravel())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv


def get_vertices_per_edge(mesh_v, mesh_f):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()"""

    vc = sp.coo_matrix(get_vert_connectivity(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:, 0] < result[:, 1]]  # for uniqueness

    return result


def vertex_quadrics(mesh):
    """Computes a quadric for each vertex in the Mesh.

    Returns:
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

    # Allocate quadrics
    v_quadrics = np.zeros((
        len(mesh.v),
        4,
        4,
    ))

    # For each face...
    for f_idx in range(len(mesh.f)):

        # Compute normalized plane equation for that face
        vert_idxs = mesh.f[f_idx]
        verts = np.hstack((mesh.v[vert_idxs], np.array([1, 1,
                                                        1]).reshape(-1, 1)))
        u, s, v = np.linalg.svd(verts)
        eq = v[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))

        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[mesh.f[f_idx, k], :, :] += np.outer(eq, eq)

    return v_quadrics


def setup_deformation_transfer(source, target, use_normals=False):
    rows = np.zeros(3 * target.v.shape[0])
    cols = np.zeros(3 * target.v.shape[0])
    coeffs_v = np.zeros(3 * target.v.shape[0])
    coeffs_n = np.zeros(3 * target.v.shape[0])

    nearest_faces, nearest_parts, nearest_vertices = source.compute_aabb_tree(
    ).nearest(target.v, True)
    nearest_faces = nearest_faces.ravel().astype(np.int64)
    nearest_parts = nearest_parts.ravel().astype(np.int64)
    nearest_vertices = nearest_vertices.ravel()

    for i in range(target.v.shape[0]):
        # Closest triangle index
        f_id = nearest_faces[i]
        # Closest triangle vertex ids
        nearest_f = source.f[f_id]

        # Closest surface point
        nearest_v = nearest_vertices[3 * i:3 * i + 3]
        # Distance vector to the closest surface point
        dist_vec = target.v[i] - nearest_v

        rows[3 * i:3 * i + 3] = i * np.ones(3)
        cols[3 * i:3 * i + 3] = nearest_f

        n_id = nearest_parts[i]
        if n_id == 0:
            # Closest surface point in triangle
            A = np.vstack((source.v[nearest_f])).T
            coeffs_v[3 * i:3 * i + 3] = np.linalg.lstsq(A, nearest_v,
                                                        rcond=-1)[0]
        elif n_id > 0 and n_id <= 3:
            # Closest surface point on edge
            A = np.vstack((source.v[nearest_f[n_id - 1]],
                           source.v[nearest_f[n_id % 3]])).T
            tmp_coeffs = np.linalg.lstsq(A, target.v[i], rcond=-1)[0]
            coeffs_v[3 * i + n_id - 1] = tmp_coeffs[0]
            coeffs_v[3 * i + n_id % 3] = tmp_coeffs[1]
        else:
            # Closest surface point a vertex
            coeffs_v[3 * i + n_id - 4] = 1.0

    matrix = sp.csc_matrix((coeffs_v, (rows, cols)),
                           shape=(target.v.shape[0], source.v.shape[0]))
    return matrix


def qslim_decimator_transformer(mesh, factor=None, n_verts_desired=None):
    """Return a simplified version of this mesh.

    A Qslim-style approach is used here.

    :param factor: fraction of the original vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: new_faces: An Fx3 array of faces, mtx: Transformation matrix
    """

    if factor is None and n_verts_desired is None:
        raise Exception('Need either factor or n_verts_desired.')

    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh.v) * factor)

    Qv = vertex_quadrics(mesh)

    # fill out a sparse matrix indicating vertex-vertex adjacency
    # from psbody.mesh.topology.connectivity import get_vertices_per_edge
    vert_adj = get_vertices_per_edge(mesh.v, mesh.f)
    # vert_adj = sp.lil_matrix((len(mesh.v), len(mesh.v)))
    # for f_idx in range(len(mesh.f)):
    #     vert_adj[mesh.f[f_idx], mesh.f[f_idx]] = 1

    vert_adj = sp.csc_matrix(
        (vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])),
        shape=(len(mesh.v), len(mesh.v)))
    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)
        result = {
            'destroy_c_cost': destroy_c_cost,
            'destroy_r_cost': destroy_r_cost,
            'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
            'Qsum': Qsum
        }
        return result

    # construct a queue of edges with costs
    queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)['collapse_cost']
        heapq.heappush(queue, (cost, (r, c)))

    # decimate
    collapse_list = []
    nverts_total = len(mesh.v)
    faces = mesh.f.copy()
    while nverts_total > n_verts_desired:
        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)
        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
            continue
        else:

            # update old vert idxs to new one,
            # in queue and in face list
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            which1 = [
                idx for idx in range(len(queue))
                if queue[idx][1][0] == to_destroy
            ]
            which2 = [
                idx for idx in range(len(queue))
                if queue[idx][1][1] == to_destroy
            ]
            for k in which1:
                queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
            for k in which2:
                queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']

            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            # remove degenerate faces
            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            faces = faces[faces_to_keep, :].copy()

        nverts_total = (len(np.unique(faces.flatten())))

    new_faces, mtx = _get_sparse_transform(faces, len(mesh.v))
    return new_faces, mtx # check mtx (= D[i] = down_transform_list[i] bellow)


def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij),
                        shape=(len(verts_left), num_original_verts)) # note len(verts_left) in  shape=(len(verts_left), num_original_verts)
    #print('in qslim_decimator_transformer, _get_sparse_transform; len(verts_left) = ', len(verts_left))
    return (new_faces, mtx)  # check mtx (= D[i] = down_transform_list[i] bellow)



def generate_transform_matrices1(mesh, factors):
    """Generates len(factors) meshes, each of them is scaled by factors[i] and
       computes the transformations between them.

    Returns:
       M: a set of meshes downsampled from mesh by a factor specified in factors.
       A: Adjacency matrix for each of the meshes
       D: csc_matrix Downsampling transforms between each of the meshes
       U: Upsampling transforms between each of the meshes
       F: a list of faces
    """

    factors = map(lambda x: 1.0 / x, factors)
    M, A, D, U, F, V = [], [], [], [], [], []
    F.append(mesh.f)  # F[0]
    V.append(mesh.v)
    A.append(get_vert_connectivity(mesh.v, mesh.f).astype('float32'))  # A[0]
    M.append(mesh)  # M[0]

    for factor in factors:
        ds_f, ds_D = qslim_decimator_transformer(M[-1], factor=factor)
        D.append(ds_D.astype('float32'))
        new_mesh_v = ds_D.dot(M[-1].v)
        #print('new_mesh_v.shape = ', new_mesh_v.shape)
        new_mesh = Mesh(v=new_mesh_v, f=ds_f)
        F.append(new_mesh.f)
        V.append(new_mesh.v)
        M.append(new_mesh)
        A.append(
            get_vert_connectivity(new_mesh.v, new_mesh.f).astype('float32'))
        #U.append(setup_deformation_transfer(M[-1], M[-2]).astype('float32'))

    return M, A, D, U, F, V 



#from spiralconv import SpiralConv

class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        #print('in SpiralConv forward before applying layer: x.shape = ', x.shape)
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value#.T
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform).T
        out = F.elu(self.conv(out))
        return out


class SpiralnetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform):
        super(SpiralnetEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels # list [a, b, c, d] for respective number of channels accross all layers, [32, 32, 32, 64] default 
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.num_vert = self.down_transform[-1].size(0)

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx]))
            else:
                self.en_layers.append(
                    SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx]))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_channels))

        
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x


    def forward(self, x):#, *indices):
        out = self.encoder(x)
        return out
    


def instantiate_model(nr_layers = 8, output_dim = 16):    
    
    device = 'cpu'

    class Mesh2:
        def __init__(self, v, f):
            self.v = v
            self.f = f

    faces = [
        # Face
        (0, 1, 23), (0, 4, 24), (1, 2, 23), (2, 3, 23), (3, 7, 23), (4, 5, 24), (5, 6, 24), (6, 8, 24), (9, 10, 24),
        
        # Upper body, arms and legs
        (11, 12, 23), (12, 14, 24), (14, 16, 24), (16, 18, 24), (16, 20, 24), (18, 20, 24), (16, 22, 24),
        (11, 13, 23), (13, 15, 23), (15, 17, 23), (15, 21, 23), (15, 19, 23), (17, 19, 23), (11, 23, 24), (12, 24, 23),
        (23, 24, 11), (24, 26, 23), (26, 28, 24), (28, 30, 24), (28, 32, 24), (30, 32, 24), (23, 25, 24), (25, 27, 23), (27, 29, 23),
        (27, 31, 23), (29, 31, 23)
        
    ]

    faces = np.array(faces)

    vertices = np.random.randn(33, 3)

    mesh_obj = Mesh2(v=vertices, f=faces)

    out_channels = []
    dilation = []
    seq_length = []
    for i in range(nr_layers - 1):
        out_channels.append(33)
        dilation.append(1)
        seq_length.append(1)

    M, A, D, _, F_, V = generate_transform_matrices1(mesh_obj, dilation) # [2, 2, 2, 1] works validated

    tmp = {
            'vertices': V,
            'face': F_,
            'adj': A,
            'down_transform': D
        }

    #dilation = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    #seq_length = [1, 1, 1, 1, 1, 1, 1, 1]

    spiral_indices_list = [
        preprocess_spiral(tmp['face'][idx], seq_length[idx],
                                tmp['vertices'][idx],
                                dilation[idx]).to(device)
        for idx in range(len(tmp['face']) - 1)
    ]
    down_transform_list = [
        to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]

    in_channels = 3
    #, 64]    16, 7, 3



    latent_channels = output_dim

    model = SpiralnetEncoder(in_channels, out_channels, latent_channels, spiral_indices_list, down_transform_list)

    return model