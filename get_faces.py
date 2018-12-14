import dlib  # 人脸处理的库dlib
import numpy as np  # 数据处理的库Numpy
import cv2  # 图像处理的库OpenCv
import os  # 读写文件

# 使用dlib自带的frontal_face_detector()函数作为人脸提取器
detector = dlib.get_frontal_face_detector()

# OpenCv 调用摄像头
cap = cv2.VideoCapture(0)

# 设置视频参数
cap.set(3, 640)
cap.set(4, 480)

# 人脸截图的计数器
faces_count = 0
# 人脸种类数目的计数器
person_count = 0

# 存储人脸的文件夹
current_face_dir = 0

# 保存的路径
path_make_dir = "faces/"
path_csv = "features/"

# 判断能否进行图片保存
save_flag = 1

while cap.isOpened():
    # 480 height * 640 width
    ret, frame = cap.read()
    '''
    函数名：cap.read()
    功  能：返回两个值
           先返回一个布尔值，如果视频读取正确，则为 True，如果错误，则为 False，也可用来判断是否到视频末尾
           再返回一个值，为每一帧的图像，该值是一个三维矩阵
           通用接收方法为：
           ret,frame = cap.read();
           这样 ret 存储布尔值，frame 存储图像
    '''
    # 返回值为当前键盘按键值
    read = cv2.waitKey(1)  # 每帧数据延迟1ms,延时不能为0，否则读取的结果会是静态帧
    #img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # 人脸数 faces
    faces = detector(frame, 1)

    # 按下 'N' 新建存储人脸的文件夹
    if read == 78:
        person_count += 1
        current_face_dir = path_make_dir + "student_" + str(person_count)
        os.makedirs(current_face_dir)
        print("新建的人脸文件夹: ", current_face_dir)
        # 将人脸计数器清零
        faces_count = 0
    # 检测到人脸
    if len(faces) > 0:
        # 矩形框
        for k, d in enumerate(faces):
            # 计算矩形框大小
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())

            # 设置矩形框颜色
            if (d.right() + int(width / 2)) > 640 or (d.bottom() + int(height / 2) > 480) or (d.left() - int(width / 2) < 0) or (d.top() - int(height / 2) < 0):
                color_rectangle = (0, 0, 255)  # 红色
                save_flag = 0  # 该情况下不适合保存图片
            else:
                color_rectangle = (255, 255, 255)  # 白色
                save_flag = 1

            cv2.rectangle(frame,
                          tuple([d.left() - int(width / 2), d.top() - int(height / 2)]),
                          tuple([d.right() + int(width / 2), d.bottom() + int(height / 2)]),
                          color_rectangle, 2)

            # 根据人脸大小生成空的图像
            im_blank = np.zeros((int(height * 2), width * 2, 3), np.uint8)

            if save_flag:
                # 按下空格保存摄像头中的人脸到本地
                if read == 32:
                    faces_count += 1
                    for a in range(height * 2):
                        for b in range(width * 2):
                            im_blank[a][b] = frame[d.top() - int(height / 2) + a][d.left() - int(width / 2) + b]
                    cv2.imwrite(current_face_dir + "/img_face_" + str(faces_count) + ".jpg", im_blank)
                    print("写入本地：", str(current_face_dir) + "/img_face_" + str(faces_count) + ".jpg")

    # 按下'Esc'键退出
    if read == 27:
        break

    # 窗口显示
    # cv2.namedWindow("camera", 0) # 如果需要摄像头窗口大小可调
    cv2.imshow("camera", frame)

# 释放摄像头
cap.release()
# 删除建立的窗口
cv2.destroyAllWindows()