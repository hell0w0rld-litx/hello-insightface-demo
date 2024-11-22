"""
计算两个人脸特征向量的余弦相似度
"""
import insightface
import numpy as np
import cv2

def calc_cos_sim(feature1, feature2):
    return np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))  # 计算两个特征向量的余弦相似度

# 加载insightface模型
model = insightface.app.FaceAnalysis(root='./', allowed_modules=None, providers=['CUDAExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

# 读取图片
img1 = cv2.imread('./images/LiuDehua.jpg')  # 第一张图片路径
img2 = cv2.imread('./images/LiuDehua2.jpg')  # 第二张图片路径

# 提取人脸特征
faces1 = model.get(img1)  # 第一张图片的人脸特征
faces2 = model.get(img2)  # 第二张图片的人脸特征

# 假设每张图片只包含一个人脸，获取第一个人脸的特征向量
if len(faces1) > 0 and len(faces2) > 0:
    emb1 = faces1[0].normed_embedding  # 第一张图片的人脸特征
    emb2 = faces2[0].normed_embedding  # 第二张图片的人脸特征

    # 计算余弦相似度
    sim = calc_cos_sim(emb1, emb2)
    print(f'Cosine similarity: {sim}')
else:
    print('没有检测到人脸，请检查图片。')
