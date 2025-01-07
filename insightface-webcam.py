"""
人脸识别Webcam实时演示
"""

import os
import os.path as osp
from pathlib import Path
import cv2
import numpy as np
import insightface
import io
from typing import Union
import base64
from pydantic import BaseModel
import time
from PIL import Image, ImageDraw, ImageFont

# 设置GPU ID
gpu_id = 0
# 人脸数据库目录
face_db = 'face_db'
# 特征比较阈值
threshold = 1.24
# 检测阈值
det_thresh = 0.50
# 检测图片尺寸
det_size = (640, 640)

# 加载人脸分析模型
model = insightface.app.FaceAnalysis(root='./', allowed_modules=None, providers=['CUDAExecutionProvider'])
# 使用小模型
# model = insightface.app.FaceAnalysis(name='buffalo_s', root='./', allowed_modules=None, providers=['CUDAExecutionProvider'])

# 准备模型检测参数
model.prepare(ctx_id=gpu_id, det_thresh=det_thresh, det_size=det_size)

# 人脸特征存储文件路径
file_path = 'face_embeddings.npy'


# 定义Pydantic数据模型
class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


# 定义一个图片缓存类，用于缓存已加载的图片
class ImageCache:
    # 静态属性data，存储已加载的图片，避免重复加载
    data = {}


# 定义获取图片的函数，通过文件名获取图片的多维数组
def get_image(name, to_rgb=False):
    """
    通过文件名获取图片多维数组
    :param name: 不包含后缀的文件名
    :param to_rgb: 是否将图片转换为RGB格式
    :return: 3维ndarray，表示图片，格式(长，宽，通道数)
    """
    # 生成一个键值对，用于在缓存中检查图片是否已存在
    key = (name, to_rgb)
    # 如果图片在缓存中，直接返回缓存中的图片
    if key in ImageCache.data:
        return ImageCache.data[key]
    # 获取当前文件所在目录，并拼接出图片所在的images目录路径
    images_dir = osp.join(Path(__file__).parent.absolute(), 'images')
    # 定义支持的图片文件扩展名
    ext_names = ['.jpg', '.png', '.jpeg']
    # 初始化图片文件路径为None
    image_file = None
    # 遍历可能的扩展名，寻找存在的图片文件
    for ext_name in ext_names:
        # 拼接文件路径
        _image_file = osp.join(images_dir, "%s%s" % (name, ext_name))
        # 检查文件是否存在，若存在则保存路径并跳出循环
        if osp.exists(_image_file):
            image_file = _image_file
            break
    # 如果没有找到图片文件，抛出异常
    assert image_file is not None, '%s not found' % name
    # 使用cv2读取图片
    img = cv2.imread(image_file)
    # 如果需要转换为RGB格式，则将BGR转换为RGB
    if to_rgb:
        img = img[:, :, ::-1]
    # 将图片存入缓存
    ImageCache.data[key] = img
    # 返回图片
    return img


# 添加人脸到特征矩阵
def add_face_to_matrix(name, embedding):
    """
    添加人脸特征到特征矩阵中
    :param name: 人脸对应的姓名
    :param embedding: 人脸特征向量
    """
    # 加载已存储的人脸特征
    data = load_faces(file_path)
    # 添加新的特征
    data.append((name, embedding))
    # 保存新的特征矩阵
    np.save('./face_embeddings.npy', np.array(data, dtype=object))
    print(f"New face added as {name}.")


# 从特征矩阵中移除人脸
def remove_face_from_matrix(name):
    """
    删除人脸特征数据
    :param name:  人脸对应的姓名
    :return:
    """
    # 加载已存储的人脸特征
    data = load_faces(file_path)
    # 筛选出需要保留的特征
    new_face_features = [feature for feature in data if feature[0] != name]
    # 保存更新后的特征矩阵
    np.save('./face_embeddings.npy', np.array(new_face_features, dtype=object))


# 打印人脸特征矩阵
def print_faces():
    """
    打印人脸特征数据
    :return:
    """
    # 加载人脸特征
    face_features = load_faces(file_path)
    # 遍历并打印每个人脸特征
    for name, feature_vector in face_features:
        print(f"Name: {name}, Feature Vector: {feature_vector}")


# 加载人脸特征文件
def load_faces(file=file_path):
    """
    加载人脸特征数据
    :param file:  npy文件路径
    :return:
    """
    # 如果文件不存在，返回空列表
    if not os.path.exists(file):
        return []
    # 加载特征矩阵并返回
    return np.load(file, allow_pickle=True).tolist()


# 调用加载人脸特征文件
load_faces(file_path)


# 计算余弦相似度
def cosine_metric(feature1, feature2):
    """
    余弦相似度
    计算两个特征向量的点积，然后除以两个向量的范数乘积。
    结果是两个向量的余弦相似度，范围在[-1, 1]之间，值越大表示两个向量越相似。-1为完全不相似，1为完全相似。
    :param feature1: 特征向量1
    :param feature2: 特征向量2
    :return: 余弦相似度值
    """
    # 计算余弦相似度
    return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))


# 比较人脸特征
def feature_compare(feature1, feature2, threshold):
    """
    人脸特征比对(欧氏距离)
    :param feature1: 特征向量1
    :param feature2: 特征向量2
    :param threshold: 阈值
    :return: 是否匹配
    """
    # 计算向量差值
    diff = np.subtract(feature1, feature2)
    # 计算平方和
    dist = np.sum(np.square(diff), 1)
    # 如果距离小于阈值，返回True，否则返回False
    return dist < threshold


# 在图片上绘制中文字符
def chineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    """
    绘制中文字符
    :param img:  图片
    :param text:   文本
    :param position:  坐标 (x, y)
    :param textColor:   颜色 (B, G, R)
    :param textSize:    字体大小
    :return: 绘制中文后的图片
    """
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换为PIL格式
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("SimHei.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_name_and_box(img, is_find, max_score, name, box):
    """
    绘制人脸框和名字
    :param img:         图片
    :param is_find:     是否找到人脸，找到和没找到的颜色不同
    :param max_score:   最大相似度，用于绘制相似度条
    :param name:        名字
    :param box:         人脸框坐标
    :return:
    """
    # 转换为整数,否则报错：
    # cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'rectangle'
    # > Overload resolution failed:
    # >  - Can't parse 'pt1'. Sequence item with index 0 has a wrong type
    # >  - Can't parse 'pt1'. Sequence item with index 0 has a wrong type
    # >  - Can't parse 'rec'. Expected sequence length 4, got 2
    # >  - Can't parse 'rec'. Expected sequence length 4, got 2
    box = box.astype(int)

    if is_find:
        # 如果找到人脸，设置绿色框
        color = (0, 255, 0)
        # 在图像上绘制名字
        # cv2.putText(img, f"{name}", (box[0], box[1]), 1, 1, color)  # 有乱码问题
        img = chineseText(img, f"{name}", (box[0], box[1]))
        x = box[2]
        y = box[3]
        y1 = int(y - max_score * (box[3] - box[1]))
        # 绘制相似度条
        cv2.rectangle(img, (x, y1), (x + 10, y), color, -1)
    else:
        # 如果未找到人脸，设置黑色框
        color = (0, 0, 0)
    # 绘制人脸框
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    # 保存带有框的图片
    cv2.imwrite(os.path.join(face_db, '%s_box.png' % name), img)
    return img


# 保存人脸图片及其特征
def save_face(image, faces, user_name, embedding):
    """
    保存人脸图片及其特征
    :param image: 图片
    :param faces: 检测到的人脸
    :param user_name: 用户名
    :param embedding: 人脸特征向量
    :return: 无
    """
    # 保存原始人脸图片
    cv2.imencode('.png', image)[1].tofile(os.path.join(face_db, '%s.png' % user_name))  # 保存图片

    # 这也是一个保存人脸的地方，带有特征向量
    rimg = model.draw_on(image, faces)
    cv2.imwrite(os.path.join(face_db, '%s_feature.png' % user_name), rimg)
    # 添加人脸特征到矩阵
    add_face_to_matrix(user_name, embedding)


# 注册人脸
def register(image, user_name):
    """
    注册人脸
    :param image:    图片
    :param user_name:   姓名
    :return: 注册结果
    """
    start_time = time.time()
    faces = model.get(image)  # 检测输入图像中的人脸
    end_time = time.time()
    total_duration = end_time - start_time  # 0.08315014839172363
    print("人脸检测耗时：", total_duration)

    if len(faces) != 1:
        return '图片检测不到人脸'  # 如果检测不到人脸，返回错误信息

    # 获取检测到的第一张人脸
    face = faces[0]
    print_face_detail(face)

    embedding = faces[0].normed_embedding

    face_features = load_faces(file_path)

    if len(face_features) == 0:  # 如果人脸库为空，则直接添加人脸到库中
        print("人脸库为空，直接添加人脸到库中")
        save_face(image, faces, user_name, embedding)
    else:
        is_new_face = True
        threshold = 0.4

        start_time = time.time()
        # 比较当前人脸与库中已有特征
        for name, feature_vector in face_features:
            dist = cosine_metric(embedding, feature_vector)  # 计算两个特征向量的余弦相似度
            # 如相似度小于阈值，则认为是同一个人，则不是新人脸
            print(f"Name: {name}, Distance: {dist}, threshold: {threshold}")
            if dist > threshold:
                # 如果相似度超过阈值，则认为是已有的人脸
                print("相似度很大，当前人脸已经存在.")
                is_new_face = False
                break
        end_time = time.time()
        total_duration = end_time - start_time
        print("人脸比对耗时：", total_duration)

        if is_new_face:
            # 保存新的人脸
            save_face(image, faces, user_name, embedding)

    # 打印人脸库中的所有人脸
    print_faces()


def print_face_detail(face):
    """
    打印人脸详细信息
    :param face: 人脸对象
    :return: 无
    """
    result = dict()
    # 获取人脸属性
    result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()  # 人脸边框
    result["kps"] = np.array(face.kps).astype(np.int32).tolist()  # 人脸关键点
    result["landmark_3d_68"] = np.array(face.landmark_3d_68).astype(np.int32).tolist()  # 3D人脸关键点
    result["landmark_2d_106"] = np.array(face.landmark_2d_106).astype(np.int32).tolist()  # 2D人脸关键点
    result["pose"] = np.array(face.pose).astype(np.int32).tolist()  # 人脸姿态
    result["age"] = face.age  # 年龄
    gender = '男'
    if face.gender == 0:
        gender = '女'  # 性别
    result["gender"] = gender
    # 开始人脸识别
    embedding = face.normed_embedding
    result["embedding"] = embedding  # 人脸特征

    print('人脸框坐标：{}'.format(result["bbox"]))  # 打印人脸框坐标
    print('人脸五个关键点：{}'.format(result["kps"]))  # 打印人脸五个关键点
    print('人脸3D关键点：{}'.format(result["landmark_3d_68"]))  # 打印人脸3D关键点
    print('人脸2D关键点：{}'.format(result["landmark_2d_106"]))  # 打印人脸2D关键点
    print('人脸姿态：{}'.format(result["pose"]))  # 打印人脸姿态
    print('年龄：{}'.format(result["age"]))  # 打印年龄
    print('性别：{}'.format(result["gender"]))  # 打印性别


# 识别人脸
def recognition(image):
    """
    识别人脸
    :param image:  带有人脸的图片
    :return: 识别结果图像
    """
    user_name = ""
    is_find_face = False
    threshold = 0.4

    # 检测图像中的人脸
    start_time = time.time()
    faces = model.get(image)  # 检测输入图像中的人脸
    end_time = time.time()
    total_duration = end_time - start_time  # 0.08315014839172363
    print("人脸检测耗时：", total_duration)

    if len(faces) != 1:
        return ''

    face = faces[0]

    embedding = faces[0].normed_embedding
    # 加载人脸特征库
    face_features = load_faces(file_path)

    if len(face_features) == 0:  # 如果人脸库为空，则直接添加人脸到库中
        print("人脸库为空，无法识别人脸")
        return ""
    else:
        draw_img = image
        start_time = time.time()
        for name, feature_vector in face_features:
            dist = cosine_metric(embedding, feature_vector)  # 计算两个特征向量的余弦相似度
            # 如相似度小于阈值，则认为是同一个人，则不是新人脸
            print(f"Name: {name}, Distance: {dist}, threshold: {threshold}")
            if dist > threshold:
                # 如果相似度超过阈值，识别到人脸
                is_find_face = True
                user_name = name
                draw_img = draw_name_and_box(image, True, dist, user_name, faces[0].bbox)
                break
        end_time = time.time()
        total_duration = end_time - start_time
        print("人脸比对耗时：", total_duration)

        return draw_img


def get_face_name_list():
    face_name_list = []
    face_features = load_faces(file_path)
    for name, feature_vector in face_features:
        print(f"Name: {name}, Feature Vector: {feature_vector}")
        face_name_list.append(name)
    return face_name_list


def open_camera():
    """
    打开摄像头, 识别人脸
    :return:
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("can't open the webcam")

    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            break

        new_frame = recognition(frame)

        if new_frame is not None and isinstance(new_frame, np.ndarray):
            cv2.imshow('Webcam', new_frame)
        else:
            print("未识别")
            cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_available_cameras():
    """
    获取可用摄像头列表
    :return:
    """
    camera_list = []
    index = 0
    while True:
        # 尝试打开摄像头
        cap = cv2.VideoCapture(index)
        # 判断是否成功打开摄像头
        if cap.isOpened():
            camera_list.append(index)
            # 释放摄像头
            cap.release()
        else:
            break
        index += 1
    return camera_list


def display_available_cameras():
    """
    打印可用摄像头列表
    :return:
    """
    camera_list = get_available_cameras()
    if len(camera_list) == 0:
        print("没有找到可用摄像头")
    else:
        print("可用摄像头列表:")
        for index in camera_list:
            print(f"摄像头 {index}")


if __name__ == "__main__":


    # 注册人脸
    # img = get_image("LiuDehua")
    # register(img,"liudehua")

    # 打开摄像头
    open_camera()
