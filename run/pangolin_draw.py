#!/usr/bin/env python
# coding=utf-8

import numpy as np
import OpenGL.GL as gl
import pangolin
import time


def X(phi):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    return Rx


def Y(theta):
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    return Ry


def Z(psi):
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    return Rz


def homo(data_3xn):
    return np.append(data_3xn, np.ones((1, data_3xn.shape[1])), axis=0)


def unHomo(data_4xn):
    return data_4xn[:3, :]


def makeT(R_3x3, t_3x1):
    T = np.eye(4)
    T[:3, :3] = R_3x3
    T[:3, 3:4] = t_3x1
    return T


class DrawGraph(object):
    def __init__(self):
        pangolin.CreateWindowAndBind("Main", 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self._scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(0.1, 0.1, -2, 0, 0, 0, pangolin.AxisDirection.AxisY)
        )
        self._handler = pangolin.Handler3D(self._scam)

        self._dcam = pangolin.CreateDisplay()
        self._dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
        self._dcam.SetHandler(self._handler)

    def draw(self, poseList_nx4x4):
        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self._dcam.Activate(self._scam)

            # pangolin.glDrawColouredCube()
            # points = np.random.random((10000, 3)) *10

            gl.glPointSize(2)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawLine(np.array([[0, 0, 0], [1.0, 0.0, 0.0]]))
            gl.glColor3f(0.0, 1.0, 0.0)
            pangolin.DrawLine(np.array([[0, 0, 0], [0.0, 1.0, 0.0]]))
            gl.glColor3f(.0, 0.0, 1.0)
            pangolin.DrawLine(np.array([[0, 0, 0], [0.0, 0.0, 1.0]]))

            origin = [[0, 0, 0]]
            originHomo = homo(np.zeros((3, 1)))
            originT = np.eye(4)

            # pangolin.Draw
            for T in poseList_nx4x4:
                originT = originT.dot(T)
                gl.glColor3f(0.5, 0.75, 0.8)
                pangolin.DrawCamera(originT)
                pAfter = originT.dot(originHomo)
                p = unHomo(pAfter)
                origin.append([p[0][0], p[1][0], p[2][0]])
                gl.glColor3f(0.0, 0.8, 0.1)
                pangolin.DrawLine(origin)
            pangolin.FinishFrame()

    def t_draw(self):
        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self._dcam.Activate(self._scam)

            pangolin.glDrawColouredCube()

            # points = np.random.random((10000, 3)) *10
            gl.glPointSize(2)
            gl.glColor3f(1.0, 0.0, 0.0)
            # pangolin.DrawPoints(points)
            for theta in np.arange(0, np.pi, np.pi / 10):
                for psi in np.arange(0, np.pi * 2, np.pi / 10):
                    T = makeT(Y(theta).dot(Z(psi)), np.random.random((3, 1)) * 10)
                    pangolin.DrawCamera(T)
            time.sleep(0.4)

            # pangolin.DrawCamera(np.eye(4))

            pangolin.FinishFrame()

    def t_draw2(self):

        trajectory = [[0, -6, 6]]
        for i in range(300):
            trajectory.append(trajectory[-1] + np.random.random(3) - 0.5)
        trajectory = np.array(trajectory)

        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self._dcam.Activate(self._scam)

            # Render OpenGL Cube
            pangolin.glDrawColouredCube(0.1)

            # Draw Point Cloud
            points = np.random.random((10000, 3)) * 3 - 4
            gl.glPointSize(1)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawPoints(points)

            # Draw Point Cloud
            points = np.random.random((10000, 3))
            colors = np.zeros((len(points), 3))
            colors[:, 1] = 1 - points[:, 0]
            colors[:, 2] = 1 - points[:, 1]
            colors[:, 0] = 1 - points[:, 2]
            points = points * 3 + 1
            gl.glPointSize(1)
            pangolin.DrawPoints(points, colors)

            # Draw lines
            gl.glLineWidth(1)
            gl.glColor3f(0.0, 0.0, 0.0)
            pangolin.DrawLine(trajectory)  # consecutive
            gl.glColor3f(0.0, 1.0, 0.0)
            pangolin.DrawLines(
                trajectory,
                trajectory + np.random.randn(len(trajectory), 3),
                point_size=5)  # separate

            # Draw camera
            pose = np.identity(4)
            pose[:3, 3] = np.random.randn(3)
            gl.glLineWidth(1)
            gl.glColor3f(0.0, 0.0, 1.0)
            pangolin.DrawCamera(pose, 0.5, 0.75, 0.8)

            # Draw boxes
            poses = [np.identity(4) for i in range(10)]
            for pose in poses:
                pose[:3, 3] = np.random.randn(3) + np.array([5, -3, 0])
            sizes = np.random.random((len(poses), 3))
            gl.glLineWidth(1)
            gl.glColor3f(1.0, 0.0, 1.0)
            pangolin.DrawBoxes(poses, sizes)

            pangolin.FinishFrame()


if __name__ == "__main__":
    dg = DrawGraph()
    dg.t_draw()
    # dg.t_draw2()
    poseList_nx4x4 = []
    for i in range(10):
        poseList_nx4x4.append(makeT(np.eye(3), np.array([1, 0, 0]).reshape((3, 1))))

    dg.draw(poseList_nx4x4)
