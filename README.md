## Easy SLAM

史上最简单的SLAM啦 ^_^

## 数据集

* 使用[TUM](https://vision.in.tum.de/data/datasets/rgbd-dataset/download) 上最简单的数据集（第一个数据集`fr1/xyz`）
* 也可以点击直接下载[fr1/xyz](https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz)
* kinect RGB-D , 单目SLAM

## 依赖

* numpy, opencv
* pangolin (python version)

## 运行

* 修改 `config/config.yaml` 更新数据集路径
  1. 运行 `easy_slam.py` 生成 `trajectory.txt` 轨迹文件， 等待运行完，`pangolin`会自动绘图
  2. 运行 `pangolin_draw.py` 显示计算出来的轨迹
  3. 带`t_`开头的函数是测试函数，具有借鉴意义

## 轨迹

![](/home/qie/PycharmProjects/EasySLAM/EasySLAM-fr1xyz.png)

## TODO List

1. 使用Python进程池，一边计算，一边绘图
2. 对环境进行三维重建，自己写Robust-ICP算法
3. 使用更难的数据集，自己写numpy-G2O 图优化算法
4. 双目SLAM高并发
5. SLAM+IMU

