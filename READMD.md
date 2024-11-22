# hello-insightface-demo



本项目是[insightface](https://github.com/deepinsight/insightface)的一个使用demo，使用insightface完成了：人脸检测，人脸识别，人脸特征比对，摄像头读取人脸做人脸识别，基本的活体检测

| 文件名                          | 描述                                     |
| ------------------------------- | ---------------------------------------- |
| `hello-insightface.py`          | 基本功能使用，包括人脸检测，人脸识别     |
| `cosine_similarity.py`          | 使用余弦相似度对比两个人脸特征向量       |
| `insightface-webcam.py`         | 使用摄像头来识别人脸，并和已注册人脸对比 |
| `hello-insightface-liveness.py` | 人脸的活体检测                           |



* [1.安装](#1.安装)

* [2.测试环境](#2.测试环境)

* [3.异常](#3.异常)

* [4.其他信息描述](#4.其他信息描述)

  


## 1.安装

```sh
pip install numpy==1.26.4
pip install onnxruntime-gpu==1.18.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install opencv-contrib-python
pip install cython   # 编译源码使用
```

复制insightface库的目录python-package过来

```sh
cd python-package
python setup.py sdist bdist_wheel
pip install ./dist/insightface-0.7.3-cp310-cp310-win_amd64.whl
```


## 2.测试环境

**系统：**

Windows 11

**Python：** 

3.10.11

**CUDA运行时:**

`nvcc -V`

> nvcc -V      
> nvcc: NVIDIA (R) Cuda compiler driver
> Copyright (c) 2005-2023 NVIDIA Corporation
> Built on Tue_Aug_15_22:09:35_Pacific_Daylight_Time_2023
> Cuda compilation tools, release 12.2, V12.2.140
> Build cuda_12.2.r12.2/compiler.33191640_0

驱动支持的最高版本

`nvidia-smi`

```sh
nvidia-smi
Fri Nov  8 15:33:05 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 552.12                 Driver Version: 552.12         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2050      WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   58C    P0              8W /   40W |       0MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```



## 3.异常

`LoadLibrary failed with error 126 onnxruntime_providers_cuda.dll`

在当前测试环境下如果通过`pip install onnxruntime-gpu`,会默认安装1.20.0，并报错：

> [E:onnxruntime:Default, provider_bridge_ort.cc:1546 onnxruntime::TryGetProviderInfo_CUDA] D:\a\_work\1\s\onnxruntime\core\session\provider_bridge_ort.cc:1209 onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126 "" when trying to load "G:\workspace\idea\py\hello-insightface_1\.venv\lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"

在当前环境下，必须使用以下命令来安装

> `pip install onnxruntime-gpu==1.18.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`
>
> 
>
> // 这是onnxruntime-gpu旧的安装方法

参考：https://github.com/microsoft/onnxruntime/issues/13576

## 4.其他信息描述：

[开源AI人脸识别项目insightface(一)-使用](https://mp.weixin.qq.com/s/GXA_udgXNeUBerfTMyglVw)

[开源AI人脸识别项目insightface(二)-安装](https://mp.weixin.qq.com/s/MeFXMFvmfz3Q-GnTnQqlcw)

[开源AI人脸识别项目insightface(三)-源码学习](https://mp.weixin.qq.com/s/aalNj69XrsIoSi-EZeOrQw)

[开源AI人脸识别项目insightface(四)-人脸特征比较](https://mp.weixin.qq.com/s/qM9ICgVxUAbwGak0xqZgsQ)

[开源AI人脸识别项目insightface(五)-摄像头人脸识别](https://mp.weixin.qq.com/s/dqjdID3Qkg8OOPiDkS5lGQ)

[开源AI人脸识别项目insightface(六)-活体检测](https://mp.weixin.qq.com/s/3AaOgPCaN8OQjZzdb_WM-w)

![扫码](http://img.cdn.codethink.vip/note/2024-11/22-22-48-40-7c335a7a55c8c3fc2722f23b836c279b-2f8021.png)
