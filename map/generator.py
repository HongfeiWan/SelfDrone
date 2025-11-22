#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
from vispy import app, gloo, scene
from vispy.util.transforms import perspective, translate, rotate


# ============================================================
# 1. 生成体素城市
# ============================================================

def generate_city(width=64, depth=64, max_height=40, seed=1,
                  road_interval=(10, 20), road_width=3,
                  building_density=0.7, min_b=3, max_b=8):
    grid = np.zeros((width, depth, max_height), dtype=np.uint8)
    rng = random.Random(seed)

    # ---- Roads ----
    xs, x = [], 0
    while x < width:
        x += rng.randint(*road_interval)
        if x < width:
            xs.append(x)

    ys, y = [], 0
    while y < depth:
        y += rng.randint(*road_interval)
        if y < depth:
            ys.append(y)

    for rx in xs:
        grid[rx - road_width // 2:rx + road_width // 2 + 1, :, 0] = 1
    for ry in ys:
        grid[:, ry - road_width // 2:ry + road_width // 2 + 1, 0] = 1

    occ = grid[:, :, 0] != 0

    # ---- Buildings ----
    for i in range(width):
        for j in range(depth):
            if occ[i, j]:
                continue
            if rng.random() > building_density:
                continue

            bw = rng.randint(min_b, max_b)
            bd = rng.randint(min_b, max_b)
            if i + bw >= width or j + bd >= depth:
                continue
            if occ[i:i+bw, j:j+bd].any():
                continue

            occ[i:i+bw, j:j+bd] = True
            h = rng.randint(4, max_height - 3)
            for xi in range(i, i+bw):
                for yj in range(j, j+bd):
                    for z in range(1, h):
                        grid[xi, yj, z] = 2
                    grid[xi, yj, h] = 3  # roof

    return grid


# ============================================================
# 2. 曝露的体素
# ============================================================

def get_exposed_voxels(grid):
    w, d, h = grid.shape
    voxels = []
    for x in range(w):
        for y in range(d):
            for z in range(h):
                if grid[x, y, z] == 0:
                    continue
                exposed = False
                for nx, ny, nz in [(x-1,y,z),(x+1,y,z),(x,y-1,z),(x,y+1,z),(x,y,z-1),(x,y,z+1)]:
                    if nx<0 or ny<0 or nz<0 or nx>=w or ny>=d or nz>=h:
                        exposed = True; break
                    if grid[nx,ny,nz] == 0:
                        exposed = True; break
                if exposed:
                    voxels.append((x,y,z, grid[x,y,z]))
    return np.array(voxels, dtype=np.int32)


# ============================================================
# 3. VisPy instanced cube
# ============================================================

vertex_shader = """
#version 120
attribute vec3 a_position;
attribute vec3 a_offset;      // instance position
attribute vec4 a_color;

varying vec4 v_color;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

void main() {
    vec3 pos = a_position + a_offset;
    gl_Position = u_projection * u_view * u_model * vec4(pos, 1.0);
    v_color = a_color;
}
"""

fragment_shader = """
#version 120
varying vec4 v_color;

void main() {
    gl_FragColor = v_color;
}
"""


def make_unit_cube():
    """Return (vertices, colors)"""
    vertices = np.array([
        # bottom
        [-.5,-.5,-.5],[ .5,-.5,-.5],[ .5, .5,-.5],
        [-.5,-.5,-.5],[ .5, .5,-.5],[-.5, .5,-.5],
        # top
        [-.5,-.5, .5],[ .5,-.5, .5],[ .5, .5, .5],
        [-.5,-.5, .5],[ .5, .5, .5],[-.5, .5, .5],
        # front
        [-.5,-.5,-.5],[ .5,-.5,-.5],[ .5,-.5, .5],
        [-.5,-.5,-.5],[ .5,-.5, .5],[-.5,-.5, .5],
        # back
        [-.5, .5,-.5],[ .5, .5,-.5],[ .5, .5, .5],
        [-.5, .5,-.5],[ .5, .5, .5],[-.5, .5, .5],
        # left
        [-.5,-.5,-.5],[-.5, .5,-.5],[-.5, .5, .5],
        [-.5,-.5,-.5],[-.5, .5, .5],[-.5,-.5, .5],
        # right
        [.5,-.5,-.5],[.5, .5,-.5],[.5, .5, .5],
        [.5,-.5,-.5],[.5, .5, .5],[.5,-.5, .5],
    ], dtype=np.float32)
    return vertices


# ============================================================
# 4. 渲染类
# ============================================================

class VoxelCanvas(app.Canvas):
    def __init__(self, grid):
        app.Canvas.__init__(self, title="Voxel City", keys="interactive", size=(1200,800))

        vox = get_exposed_voxels(grid)
        print("Exposed voxels:", len(vox))

        offsets = vox[:, :3].astype(np.float32)
        colors = np.ones((len(vox), 4), dtype=np.float32)
        colors[vox[:,3] == 1] = (0.1,0.1,0.1,1)
        colors[vox[:,3] == 2] = (0.8,0.8,0.8,1)
        colors[vox[:,3] == 3] = (0.6,0.3,0.1,1)

        cube_v = make_unit_cube()

        self.program = gloo.Program(vertex_shader, fragment_shader)

        # cube vertices
        self.program["a_position"] = cube_v

        # instance attribute (Correct API)
        self.program["a_offset"] = offsets
        self.program["a_color"] = colors

        # set instancing
        gloo.set_vertex_attrib_divisor(self.program.attribute_buffers["a_offset"].buffer_id, 1)
        gloo.set_vertex_attrib_divisor(self.program.attribute_buffers["a_color"].buffer_id, 1)

        self.n_instances = len(offsets)
        self.n_vertices = len(cube_v)

        # transforms
        self.theta = 30
        self.phi = 35
        self.distance = 200

        self._timer = app.Timer("auto", connect=self.update, start=True)

        self.show()

    # ----------------------- camera -----------------------

    def on_mouse_move(self, event):
        if event.is_dragging:
            dx, dy = event.delta
            self.theta += dx * 0.5
            self.phi += dy * 0.5

    def on_mouse_wheel(self, event):
        self.distance *= 0.9 ** event.delta[1]

    # ----------------------- draw -------------------------

    def on_draw(self, event):
        gloo.clear(color="white", depth=True)

        model = np.eye(4, dtype=np.float32)
        view = np.eye(4, dtype=np.float32)

        # camera rotation
        rotate(view, self.theta, (0,1,0))
        rotate(view, self.phi, (1,0,0))
        translate(view, (0, 0, -self.distance))

        proj = perspective(45.0, self.size[0]/self.size[1], 1.0, 1000.0)

        self.program["u_model"] = model
        self.program["u_view"] = view
        self.program["u_projection"] = proj

        # Draw instanced cube mesh
        self.program.draw("triangles", instances=self.n_instances)


# ============================================================
# 5. 主入口
# ============================================================

if __name__ == "__main__":
    city = generate_city()
    canvas = VoxelCanvas(city)
    app.run()
