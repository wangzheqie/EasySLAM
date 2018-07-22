#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import yaml
import time
import copy
import cv2
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from pangolin_draw import DrawGraph
import threading


class EasySlamError(Exception):
    pass


class TxtFileEmpty(Exception):
    pass


class EasySlam(object):
    """
    Easy, Simplest RGB-D SLAM without back-end optimization
    """

    def __init__(self, config):
        """

        Parameters
        ----------
        config
        """

        self._config = config
        self.debug = self._config["IsDebug"]
        self.__images = None  # all images'  name
        self.__depths = None  # all depths' name
        # self.__currImg = None  # current frame image name
        # self.__currKeypoints = None  # current frame image keypoints
        # self.__currDescriptors = None  # current frame image descriptors

        self.__intrinsic = np.loadtxt(self._config["IntrinsicPath"])
        self.__distortion = np.loadtxt(self._config["DistortionPath"])
        self.__factor = self._config["CameraFactor"]

        # data path
        self.__rgbPath = os.path.join(self._config["DatasetPath"], self._config["ImagesPath"])
        self.__depthPath = os.path.join(self._config["DatasetPath"], self._config["DepthPath"])

        # for tracking
        self.__img1Name = None  # image name
        self.__img2Name = None
        self.__depth1Name = None
        self.__depth2Name = None
        self.__img1 = None
        self.__img2 = None  # current, = self.__currImg
        self.__kps1 = None
        self.__kps2 = None  # current, = self.__currKeypoints
        self.__despt1 = None
        self.__despt2 = None  # current, = self.__currDescriptors
        self.__pt3D1_nx3 = None  # Nx3
        self.__pt3D2_nx3 = None  # Nx2
        self.__pt2D1_nx2 = None  # Nx2
        self.__pt2D2_nx2 = None  # Nx2
        self.__disparity = None
        self.__matches12 = None
        self.__3DAnd2DCoor = None  # 3D and 2D correspondences in different frame
        self.pose_nx4x4 = []

        # system coordiante
        self.__baseCoor = np.eye(4)

    def loadTxt(self, txtFile):
        """

        Parameters
        ----------
        txtFile

        Returns
        -------

        """
        with open(txtFile, 'r') as file:
            data = file.read()
            lines = data.replace(",", " ").replace("\t", " ").split("\n")

        if len(lines) > 0:
            index = 0
            for line in lines:
                img_namei = line + self._config["ImagesExt"]
                lines[index] = img_namei
                index += 1
            return lines
        else:
            raise TxtFileEmpty

    def loadData(self):
        """

        Returns
        -------

        """
        imgNamesPath = os.path.join(self._config["DatasetPath"], self._config["ImagesName"])
        depthNamesPath = os.path.join(self._config["DatasetPath"], self._config["DepthsName"])

        self.__images = self.loadTxt(imgNamesPath)
        self.__depths = self.loadTxt(depthNamesPath)

    def featureDetect(self, imgName, num=100):
        """

        Parameters
        ----------
        imgName

        Returns
        -------

        """
        imgPath = os.path.join(self.__rgbPath, imgName)
        img = cv2.imread(imgPath)
        orb = cv2.ORB_create(num, scaleFactor=1.25, nlevels=8)
        keypoints = orb.detect(img)
        keypoints, descriptors = orb.compute(img, keypoints)
        if self.debug:
            cv2.drawKeypoints(img, keypoints, img)
            cv2.imshow("imgi", img)
            cv2.waitKey(self._config["DelayTime"])
        return img, keypoints, descriptors

    def findCorrespondences(self, num=20):
        """

        Returns
        -------

        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.__despt1, self.__despt2)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) >= num:
            matches = matches[:num]

        if self.debug:
            img = cv2.drawMatches(self.__img1, self.__kps1, self.__img2, self.__kps2, matches, outImg=None, flags=2)
            cv2.imshow("correspondences", img)
        return matches

    def getPts3DAnd2DSame(self, depthName, kps):
        """
        3D and 2D point from the same camera frame
        Parameters
        ----------
        depthName
        kps

        Returns
        -------

        """
        depthPath = os.path.join(self.__depthPath, depthName)
        depth = cv2.imread(depthPath)
        K = self.__intrinsic
        z_list = []
        x_list = []
        y_list = []

        u_list = []
        v_list = []
        for kp in kps:
            pixel = kp.pt
            z_mm = depth[int(pixel[1]), int(pixel[0])][0]
            # do not remove elements even z_mm == 0.0
            x_mm = (pixel[0] - K[0, 2]) * z_mm / K[0, 0]
            y_mm = (pixel[1] - K[1, 2]) * z_mm / K[1, 1]

            z_list.append(z_mm)
            x_list.append(x_mm)
            y_list.append(y_mm)

            u_list.append(pixel[0])
            v_list.append(pixel[1])

        if self.debug:
            cv2.imshow("depth", depth)
            cv2.waitKey(self._config["DelayTime"])

        pts3D = np.array([[x_list], [y_list], [z_list]]).reshape(3, -1).T
        pts2D = np.array([[u_list], [v_list]]).reshape(2, -1).T
        return pts3D, pts2D

    def getPts3DAnd2DCorrespondences(self):
        """
        -> self.__3DAnd2DCoor
        Returns
        -------

        """
        coor_3D = []
        coor_2D = []
        for m in self.__matches12:
            if self.__pt3D1_nx3[m.queryIdx][2] <= 0.0:
                continue
            else:
                coor_3D.append(self.__pt3D1_nx3[m.queryIdx])
                coor_2D.append(self.__pt2D2_nx2[m.trainIdx])

        coor_3D = np.array(coor_3D).reshape((len(coor_3D), -1))
        coor_2D = np.array(coor_2D).reshape((len(coor_2D), -1))

        self.__3DAnd2DCoor = [coor_3D, coor_2D]

    def makeT(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3:4] = t

        return T

    def solvePnP(self):

        # DLS better then EPNP
        # retval, rvec, tvec = cv2.solvePnP(self.__pt3D1_nx3, self.__pt2D1_nx2, self.__intrinsic, self.__distortion, cv2.SOLVEPNP_EPNP)
        pt3D = self.__3DAnd2DCoor[0]
        pt2D = self.__3DAnd2DCoor[1]

        retval, rvec, tvec, inliers = cv2.solvePnPRansac(pt3D, pt2D, self.__intrinsic,
                                                         self.__distortion, flags=cv2.SOLVEPNP_DLS)
        R, jacobian = cv2.Rodrigues(rvec)

        return self.makeT(R, tvec)

    def showGraphMatplotlib(self, T):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.scatter(0, 0, 0, c='b', marker='^')

    def showGraphPangolin(self, pose_nx4x4):
        dg = DrawGraph()
        dg.draw(pose_nx4x4)

    def run(self):
        self.loadData()

        # if self.debug:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.set_xlabel("X")
        #     ax.set_ylabel("Y")
        #     ax.set_zlabel("Z")
        #     ax.scatter(0, 0, 0, c='b', marker='^')
        # pose_nx4x4 = []

        for i in range(0, len(self.__images) - 10, 1):
            print("now process the " + str(i) + " image...")
            self.__img1Name = self.__images[i]
            self.__img2Name = self.__images[i + 1]
            self.__depth1Name = self.__depths[i]
            self.__depth2Name = self.__depths[i + 1]

            self.__img1, self.__kps1, self.__despt1 = self.featureDetect(self.__img1Name, num=self._config["DetectNum"])
            self.__img2, self.__kps2, self.__despt2 = self.featureDetect(self.__img2Name, num=self._config["DetectNum"])

            self.__matches12 = self.findCorrespondences(num=self._config["CorrNum"])

            self.__pt3D1_nx3, self.__pt2D1_nx2 = self.getPts3DAnd2DSame(self.__depth1Name, self.__kps1)
            self.__pt3D2_nx3, self.__pt2D2_nx2 = self.getPts3DAnd2DSame(self.__depth2Name, self.__kps2)

            self.getPts3DAnd2DCorrespondences()

            T = self.solvePnP()
            self.pose_nx4x4.append(T)

        return self.pose_nx4x4

        # plt.show()


def loadYaml(file):
    with open(file, 'r') as stream:
        try:
            data = yaml.load(stream)
        except yaml.YAMLError as e:
            print(e)
    return data


def t_after_run():
    config = loadYaml("../config/config.yaml")
    slam = EasySlam(config)
    pose_nx4x4 = slam.run()
    trajatory = np.array(pose_nx4x4).reshape((-1, 1, 16)).reshape((-1, 16))
    np.savetxt("trajectory.txt", trajatory)
    dg = DrawGraph("Draw")
    dg.draw(pose_nx4x4)


def t_thread_pool():
    config = loadYaml("../config/config.yaml")
    pose_nx4x4 = []
    slam = EasySlam(config)
    dg = DrawGraph("Draw")

    slamLock = threading.Lock()
    drawLock = threading.Lock()

    def drawGraph():
        global pose_nx4x4, slam, dg
        drawLock.acquire()
        dg.draw(slam.pose_nx4x4)
        slamLock.release()

        drawGraph()

    def runSlam():
        global pose_nx4x4, slam, dg
        slamLock.acquire()
        slam.run()
        drawLock.release()
        runSlam()

    slamLock.acquire()

    t1 = threading.Thread(target=runSlam)
    t2 = threading.Thread(target=drawGraph)
    t1.start()
    t2.start()
    t1.join()
    t2.join()


def t_multi_thread():
    class slamThread(threading.Thread):
        def __init__(self, threadID, name):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.name = name

        def run(self):
            threadLock.acquire()
            print("now solve ... ")
            pose_nx4x4 = slam.run()
            print("now draw ... ")
            dg.draw(pose_nx4x4)
            threadLock.release()

            threadLock.acquire()
            dg.t_draw()
            dg.t_draw2()
            threadLock.release()

    threadLock = threading.Lock()

    threads = []
    thread_solve = slamThread(1, "solve")
    thread_draw = slamThread(2, "draw")
    thread_solve.start()
    thread_draw.start()
    threads.append(thread_solve)
    threads.append(thread_draw)

    for t in threads:
        t.join()


if __name__ == "__main__":
    t_after_run()
