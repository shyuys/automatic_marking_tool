# coding=utf-8
# Description : 2021.03.01 测试数据
# ==== 屏蔽tensorflow输出的log信息 ====
# 注意：代码在import tensorflow之前
# 代码运行的配置环境需要描述
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import time
import cv2
import os
import shutil
import csv
import numpy as np
import tensorflow as tf
dnn_dir=os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dnn_dir,"model/tf2"))

import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOv3, decode


# ==== 解决cudnn初始化失败的bug ====
# 2021-02-26 15:33:01.420905: E tensorflow/stream_executor/cuda/cuda_dnn.cc:328]
# Could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED
# GPU内存占用太多，导致没有合理分配内存。需要使用tf.config()函数。
# 获取当前主机的特定运算设备列表
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
cpus = tf.config.experimental.list_physical_devices(device_type="CPU")
print("查询gpu设备：", gpus)
print("查询cpu设备：", cpus)
'''
默认情况下，tensorflow将使用几乎所有可用的显存，以避免内存碎片化带来的性能损失。
但是tensorflow提供了两种显存使用策略，让我们能够更灵活地控制程序的显存使用方式：
-1、仅在需要时申请显存空间(程序初始运行时消耗很少的显存，随着程序的运行而动态申请显存)；
-2、限制消耗固定大小的显存(程序不会超出限定的显存大小，若超出就报错)
#设置仅在需要时申请显存空间；
'''
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def BuildModel(weightsDir, INPUT_SIZE):
    '''
    构建模型
    请后续补充完善 ... ...
    :param weightsDir:
    :param INPUT_SIZE:
    :return:
    '''
    # 确定模型输入
    input_layer = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
    # 确定模型输出
    feature_maps = YOLOv3(input_layer)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)
    # 构建模型
    model = tf.keras.Model(input_layer, bbox_tensors)
    # 加载模型参数
    model.load_weights(weightsDir)
    return model


# //////////////////////////////////////////////////////////////////////////////////////////
# def resize_for_inference(image,model):
#     # 将图片补全成正方形，然后resize成416*416
#     resize_method = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]#
#     img_square=pic_process.makeBorder(image)#
#     # 图像缩放，缩放到416*416大小
#     img_416 = cv2.resize(img_square, (416, 416), interpolation=resize_method[1])
#     # img_416=img_square
#     return img_416

def inference_and_bbox(frame,labels_list,model,INPUT_SIZE,SCORE_THRESHOLD=0.6,IOU_THRESHOLD=0.5):
    '''
    输出格式[[751, 152, 423, 514, 'Dry_garbage'], [15, 176, 380, 672, 'Dry_garbage']]

    :param frame:
    :param model:
    :param INPUT_SIZE:
    :param SCORE_THRESHOLD:
    :param IOU_THRESHOLD:
    :return:
    '''
    # Predict Process
    # 第二步：图像预处理
    # 具体：图像比例缩放-填充-均一化

    image_size = frame.shape[:2]
    image_data = utils.image_preporcess(np.copy(frame), [INPUT_SIZE, INPUT_SIZE])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    # 第三步：模型预测输出
    # 方法：model.predict
    # 具体：将输出进行格式转换
    pred_bbox = model.predict(image_data)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]

    pred_bbox = tf.concat(pred_bbox, axis=0)

    # 第四步：确定预测框的信息
    # 具体：确定预测框在原始图片位置--确定超出边界的预测框索引-确定分数大于一定阈值的预测框索引--确定概率最大所对应类别索引
    #       --确定满足三个索引的预测框  其中分数=置信值*分类最大概率值
    # bboxes格式为[预测框的数量，预测框位置+分数+类别] shape为[-1,6]
    bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, SCORE_THRESHOLD)

    # 第五步：预测框冗余处理
    # 具体：找出拥有该类别最大分数的预测框-存储该预测框-计算该预测框与其他预测框的Iou-根据Iou相关条件删除预测框
    #       -剩下的预测框继续执行上述四个步骤，直至没有预测框
    bboxes = utils.nms(bboxes, IOU_THRESHOLD, method='nms')

    #################################################
    # 第六步：将预测框打标并且返回
    # 框框显示颜色，每一类有不同的颜色，每种颜色都是由RGB三个值组成的，所以size为(len(labels), 3)
    np.random.seed(537)
    COLORS = np.random.randint(0, 255, size=(len(labels_list),3), dtype="uint8")


    req_dict = {}

    image_h, image_w = image_size
    box_num = len(bboxes)
    req_dict["box_num"] = box_num
    object_list = []
    for i, each_bbox_float64 in enumerate(bboxes):
        each_bbox_int32 = np.array(each_bbox_float64[:4], dtype=np.int32)

        x1 = int(each_bbox_int32[0])
        y1 = int(each_bbox_int32[1])
        x2 = int(each_bbox_int32[2])
        y2 = int(each_bbox_int32[3])

        score = each_bbox_float64[4]
        classID=int(each_bbox_float64[5])
        w = x2 - x1
        h = y2-y1

        # 下面是画框，线条粗细为2px
        color_bgr = [int(c) for c in COLORS[classID]]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        cv2.rectangle(frame, (x1, y1), (x2,y2), color_bgr,bbox_thick)  # 线条粗细为2px

        # 下面是将预测信息写到图片上
        label=labels_list[classID]
        text = '%s: %.2f' % (label, score)
        fontScale = 1
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, color_bgr, bbox_thick // 2, lineType=cv2.LINE_AA)

        if label not in req_dict:
            req_dict[label] = 1
        else:
            req_dict[label] += 1
        object_list.append([x1, y1, h, w, label])
    req_dict["object_list"] = object_list
    return frame, req_dict

def sort_fileName(picDir):
    '''
    不用匿名函数，事方便调试
    :param picDir:
    :return:
    '''
    temp = picDir.split("_")[2].replace(".jpeg", "")
    return eval(temp)


if __name__ == "__main__":
    test_handle = 3

    if test_handle == 1:
        # 测试一张图片

        print(">>>测试单张图进行推理。")

        # 读取原始图片
        image_path = "./test_picture/20210602-003648_DS1_8.jpeg"

        image = cv2.imread(image_path)  # 读图像，读进来直接是BGR 格式数据格式在 0~255
        # import sys
        # sys.exit()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像转换为RGB
        #
        # image=resize_for_inference(image)
        # image=image/255
        result = inference_and_bbox(frame, model, INPUT_SIZE, SCORE_THRESHOLD=0.6, IOU_THRESHOLD=0.5)
        result = inference_and_bbox(image)
        cv2.namedWindow("result", 0)
        # cv2.resizeWindow("result", 1920 // 2, 1080 // 2)  # 该表窗体的宽、高；
        cv2.imshow("result", result)
        cv2.waitKey(0)

    elif test_handle == 2:
        # 测试一个视频

        print(">>>测试单个视频进行推理。")

        video_dir = "./EEJG4231.MP4"  # 捕获视频的路径
        cap = cv2.VideoCapture(video_dir)
        fps = cap.get(5)  # 获得帧速率
        width = int(cap.get(3))  # 获得帧的宽度
        high = int(cap.get(4))  # 获得帧的高度
        print("帧速: {0}；帧高: {1}；帧宽: {2}".format(fps, high, width))
        # 检查是否初始化捕获
        if not cap.isOpened():
            print("Cannot open camera!")
            exit()

        # 保存视频
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 指定编码器;即视频要存储的格式
        out = cv2.VideoWriter('demo_10epochs.mp4', fourcc, 60.0, (width, high), True)  # 视频存储，width, high一定为整型数据

        while cap.isOpened():
            # 逐帧捕获
            ret, frame = cap.read()
            if not ret:  # 如果正确读取帧，ret为True；否则为False
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将图像转换为RGB
            result = inference_and_bbox(image)
            out.write(result)  # 将捕捉到的图像存储，注意尺寸与VideoWriter 中指定的尺寸一致。

            cv2.namedWindow("result", 0)
            # cv2.resizeWindow("result", 1920 // 4, 1080 // 4)  # 该表窗体的宽、高；
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 键盘监听，间隔1ms，监听到'q'
                break  # 跳出while循环体

        cap.release()  # 完成所有操作后，释放捕获器
        out.release()  # 释放 打开视频存储占用的资源
        cv2.destroyAllWindows()  # 关闭窗口并取消分配任何相关的内存使用

    elif test_handle == 3:
        # 测试一个图片文件夹SCORE_THRESHOLD=0.6,IOU_THRESHOLD=0.5

        print(">>>测试单个图片文件夹进行推理。")
        folder = "./test_picture"

        # ------   构建模型   -------
        # 输入模型的地址
        dnn_dir = os.path.dirname(os.path.abspath(__file__))
        weightsDir = os.path.join(dnn_dir,"model/tf2/yolov3.h5")
        # 输入模型时图片的尺寸
        INPUT_SIZE = 416
        model = BuildModel(weightsDir, INPUT_SIZE)

        # 得到labels_list列表，如 ['Dry_garbage']
        labelsPath = "model/tf2/data/classes/ashcan.names"
        with open(labelsPath, 'rt') as f:
            labels_list = f.read().rstrip('\n').split('\n')

        # ------ 配置推理阈值   -------
        # 是目标的概率
        SCORE_THRESHOLD = 0.6
        # 目标和背景的阈值
        IOU_THRESHOLD = 0.5

        # ------    设置合成视频的帧速、帧宽、帧高   --------
        fps, width, high = 10.0, 1920, 1080
        print("帧速: {0}；帧高: {1}；帧宽: {2}".format(fps, high, width))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 指定编码器;即视频要存储的格式
        out = cv2.VideoWriter('ashcan_5epochs.avi', fourcc, fps, (width, high), True)  # 视频存储，width, high一定为int

        ficNameList = os.listdir(folder)
        # 必须做排序
        # 先过滤掉不是jpeg的图片
        filterList = list(set([i if i.endswith(".jpeg") else "" for i in ficNameList]))

        if "" in filterList:
            filterList.remove("")

        filterList.sort(key=sort_fileName)
        # 匿名函数，出了错，不好调试
        # ficNameList.sort(key=lambda picDir:picDir.split("_")[2].replace(".jpeg",""))
        print(filterList)

        for fileName in filterList:
            if fileName.endswith(".jpeg"):
                image_path = os.path.join(folder, fileName)

                image = cv2.imread(image_path)  # 读图像，读进来直接是BGR 格式数据格式在 0~255
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像转换为RGB

                T1 = time.perf_counter()
                frame, inferResult = inference_and_bbox(image,labels_list,model, INPUT_SIZE, SCORE_THRESHOLD, IOU_THRESHOLD)
                print("inferResult=",inferResult)

                T2 = time.perf_counter()
                # 将捕捉到的图像存储，注意尺寸与VideoWriter 中指定的尺寸一致。
                out.write(frame)

                # ------------------------------------------
                # 写入csv 应急；后续再修改脚本
                # 内容列表 2021.06.10
                # 这里还有bug,就是最后一行数据无法写入进去--备注：
                #
                picName = fileName
                garbageNum = inferResult["box_num"]
                stu1 = [picName, garbageNum]

                csvDir = "20210611_13601096.csv"
                csv_handle = open(csvDir, 'a', newline='')  # 打开文件，追加a

                csv_write = csv.writer(csv_handle, dialect='excel')  # 设定写入模式
                csv_write.writerow(stu1)  # 写入具体内容
                # ---------------------------------------------------------------
                cv2.namedWindow("show", 0)
                cv2.imshow("show", frame)
                cv2.waitKey(0)

        # 释放 打开视频存储占用的资源
        out.release()
        # 关闭窗口并取消分配任何相关的内存使用
        cv2.destroyAllWindows()