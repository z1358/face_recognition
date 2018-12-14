# -*- coding: utf-8 -*-
import dlib  # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2  # 图像处理的库 OpenCv
import pandas as pd  # 数据处理的库 Pandas
import os

# 人脸识别模型,Dlib 检测器和预测器
facerec = dlib.face_recognition_model_v1("model/2.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/1.dat')
path_faces = "faces/"
# 字体
font = cv2.FONT_HERSHEY_SIMPLEX
# 定义空的list存放人脸的标签
label = []
# 计算两个向量间的欧式距离
# 存储所有标签
all = []


def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print("e_distance: ", dist)

    if dist > 0.5:
        return "other"
    else:
        return "same"


faces = os.listdir(path_faces)
for person in faces:
    label.append(person)  # 保存标签
# 处理存放所有人脸特征的 CSV
path_features_known_csv = "features_all.csv"
csv_rd = pd.read_csv(path_features_known_csv, header=None)

# 用来存放所有录入人脸特征的数组
features_before = []

# 读取已知人脸数据
# known faces
for i in range(csv_rd.shape[0]):
    features_someone_arr = []
    for j in range(0, len(csv_rd.ix[i, :])):
        features_someone_arr.append(csv_rd.ix[i, :][j])
    #    print(features_someone_arr)
    features_before.append(features_someone_arr)
print("Faces in Database：", len(features_before))

# 创建 cv2 摄像头对象
cap = cv2.VideoCapture(0)

# 设置视频参数
cap.set(3, 640)
cap.set(4, 480)

# cap.isOpened() 返回 true/false 检查初始化是否成功
while cap.isOpened():
    flag, img_rd = cap.read()
    read = cv2.waitKey(1)

    # 取灰度
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    # 人脸数 faces
    faces = detector(img_gray, 1)

    # 存储所有人脸的名字
    pos_namelist = []
    name_namelist = []

    # 按下Esc键退出
    if read == 27:
        break
    else:
        # 检测到人脸
        if len(faces) != 0:
            # 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
            features_now = []
            for i in range(len(faces)):
                shape = predictor(img_rd, faces[i])
                features_now.append(facerec.compute_face_descriptor(img_rd, shape))

            # 遍历捕获到的图像中所有的人脸
            for k in range(len(faces)):
                # 让人名跟随在矩形框的下方
                # 确定人名的位置坐标
                # 先默认所有人不认识，是 unknown
                name_namelist.append("unknown")

                # 每个捕获人脸的名字坐标
                pos_namelist.append(
                    tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                # 对于某张人脸，遍历所有存储的人脸特征
                for i in range(len(features_before)):
                    print("with person_", str(i + 1), "the ", end='')
                    # 将某张人脸与存储的所有人脸数据进行比对
                    compare = return_euclidean_distance(features_now[k], features_before[i])
                    if compare == "same":  # 找到了相似脸
                        name_namelist[k] = label[i]
                        all.append(name_namelist[k])

                # 矩形框
                for k, d in enumerate(faces):
                    # 绘制矩形框
                    cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)

            # 在人脸框下面写人脸名字
            for i in range(len(faces)):
                cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

    print("Name list now:", name_namelist, "\n")

    cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # 窗口显示
    cv2.imshow("camera", img_rd)

# 释放摄像头并删除建立的窗口
cap.release()
cv2.destroyAllWindows()

# 去除重复, 同时保持先后顺序不变
all_unique = sorted(set(all), key=all.index)
# print(all)
# print(all_unique)
with open('name.txt', 'a') as f:
    for i in all_unique:
        f.write(i + '\n')