# 导入OpenCV库，用于图像处理
import cv2
# 导入insightface库中的FaceAnalysis模块，用于人脸分析
from insightface.app import FaceAnalysis
# 导入os.path中的方法，用于处理路径
import os.path as osp
# 导入Path类，用于路径操作
from pathlib import Path

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


# 初始化FaceAnalysis应用程序，指定模型目录和计算提供者
# root: 模型的目录，默认是~/.insightface；这个地方指定为当前目录的
# allowed_modules: 允许的模块（此处为None表示默认设置）
# providers: 指定的计算提供者（此处为CUDA，以便在支持的设备上加速），使用GPU加速,如果CUDA环境没有安装，则会自动切换到CPU计算
app = FaceAnalysis(root='./', allowed_modules=None, providers=['CUDAExecutionProvider'])
# 准备模型，设置计算设备为GPU和人脸检测的大小
app.prepare(ctx_id=0, det_size=(640, 640))
# 调用get_image函数获取图片的多维数组，图片名为'LiuDehua'，不带扩展名
# 图片应放在./images目录下
img = get_image('LiuDehua')
# 使用FaceAnalysis应用程序检测人脸，返回人脸特征信息列表（每个元素代表一张人脸）
faces = app.get(img)
# 打印检测到的所有人脸信息
print("faces : ", faces)
# 打印检测到的人脸数量
print("len : ", len(faces))
# 在图片上绘制检测到的人脸特征，并返回绘制后的图片
out_img = app.draw_on(img, faces)

# 将处理后的图片保存到本地，文件名为out_put.jpg
cv2.imwrite("./out_put.jpg", out_img)
# 显示处理后的图片窗口，窗口标题为“frame”
cv2.imshow("frame", out_img)

# 等待键盘输入，按下“Q”键关闭窗口
if cv2.waitKey(0) & 0xFF == ord('Q'):
    # 销毁所有窗口
    cv2.destroyAllWindows()
