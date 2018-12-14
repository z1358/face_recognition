# features()                      获取某张图像的特征
# write_into_csv()                获取某个路径下所有图像的特征，并写入 CSV
# compute_the_mean()              从 CSV中读取特征，并计算特征均值
import cv2
import os
import dlib
from skimage import io
import csv
import numpy as np
import pandas as pd

path_faces = "faces/"
path_feature = "features/"
# 存放所有特征均值的 CSV 的路径
path_csv_feature_all = "features_all.csv"
# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# dlib 68 点特征预测器1.dat即shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor('model/1.dat')

# Dlib 人脸识别模型 2.dat即dlib_face_recognition_resnet_model_v1.dat
facerec = dlib.face_recognition_model_v1("model/2.dat")

label = []  # 定义空的list存放人脸的标签


def features(path_img):  # 返回单张图像的特征
    img = io.imread(path_img)
    my_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # opencv默认的彩色图像的颜色空间是BGR
    faces = detector(my_img, 1)
    print("检测到人脸的图像：", path_img, "\n")
    if len(faces) != 0:
        shape = predictor(my_img, faces[0])
        face_descriptor = facerec.compute_face_descriptor(my_img, shape)
    else:
        face_descriptor = 0
    return face_descriptor

# 将文件夹中照片特征提取出来，写入 CSV
#   path_faces_personX:     图像文件夹的路径
#   path_feature:           要生成的 CSV 路径
def write_into_csv(path_faces_personX, path_feature):
    dir_pics = os.listdir(path_faces_personX)
    with open(path_feature, "w", newline="") as csvfile:
        write_place = csv.writer(csvfile)
        for i in range(len(dir_pics)):
            # 调用features()得到特征
            print("正在读的人脸图像：", path_faces_personX + "/" + dir_pics[i])
            feature = features(path_faces_personX + "/" + dir_pics[i])
            #  print(feature)
            # 遇到没有检测出人脸的图片跳过
            if feature == 0:
                i += 1
            else:
                write_place.writerow(feature)


# 读取某人所有的人脸图像的数据，写入 person.csv
faces = os.listdir(path_faces)
for person in faces:
    print("Now the person is " + person )
    print("Now the features are from " + path_feature + person + ".csv")
    write_into_csv(path_faces + person, path_feature + person + ".csv")
    label.append(person)  # 保存标签


# 从 CSV 中读取数据，计算特征的均值
def compute_the_mean(path_csv_rd):
    column_names = []
    # 128列特征
    for feature_num in range(128):
        column_names.append("features_" + str(feature_num + 1))

    # 利用pandas读取csv
    rd = pd.read_csv(path_csv_rd, names=column_names)

    # 存放128维特征的均值
    feature_mean = []

    for feature_num in range(128):
        tmp_arr = rd["features_" + str(feature_num + 1)]
        tmp_arr = np.array(tmp_arr)

        # 计算某一个特征的均值
        tmp_mean = np.mean(tmp_arr)
        feature_mean.append(tmp_mean)
    return feature_mean


with open(path_csv_feature_all, "w", newline="") as csvfile:
    write_place = csv.writer(csvfile)
    csv_rd = os.listdir(path_feature)

    for i in range(len(csv_rd)):
        feature_mean = compute_the_mean(path_feature + csv_rd[i])
        # print(feature_mean)
        print("得到的特征均值被存放在：")
        print(path_csv_feature_all)
        write_place.writerow(feature_mean)