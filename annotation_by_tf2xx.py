# -*- coding: utf-8 -*-
# jn10010537
# 通过tensorflow 2.xx的yolov3的模型进行自动打标
# 通过读取配置文件来获得参数
import os
import sys
import time
import cv2
import tf2xx_judge as dnn  # dnn模块
import creatXml            # 标注xml的模块
from config import load_json_config

def file_check_from_folder(folderDir,endswithList):
    '''
    要求：文件夹下只有文件，并且文件后缀都在支持的后缀列表里；
    输入：文件夹的路径，以及允许其中文件的后缀列表；

    输出： 是否满足要求，文件路径列表以及不是文件的列表、文件后缀不支持的列表
    :param folderDir: 文件夹的路径
    :param endswithList: 支持的文件后缀列表
    :return:
    '''
    # 判断是否是目录
    assert os.path.isdir(folderDir), '不是文件夹目录或者不存在！【20210709-01】'
    # 获得文件路径的列表
    picDirList = [os.path.join(folderDir, picName) for picName in os.listdir(folderDir)]

    is_supported = True
    not_right_file_list = []
    not_right_extension_list = []
    for picDir in picDirList:
        # 判定是否为文件(且文件存在）
        if not os.path.isfile(picDir):
            is_supported = False
            not_right_file_list.append(picDir)
        else:
            (folder, filename_extension) = os.path.split(picDir)
            (filename, extension) = os.path.splitext(filename_extension)
            if extension not in endswithList:
                is_supported = False
                not_right_extension_list.append(picDir)
    return is_supported,picDirList,not_right_file_list,not_right_extension_list


# 加载图片文件夹，进行自动生成xml文件；
def generateXml(picDirList,labelDir,weightsDir,INPUT_SIZE,SCORE_THRESHOLD,IOU_THRESHOLD,show_handle=False):
    # ---------------------   1、读取标签列表   ------------------------
    # 得到labels_list列表，如 ['Dry_garbage']
    with open(labelDir, 'rt') as f:
        labels_list = f.read().rstrip('\n').split('\n')


    # ---------------------   2、构建模型   ------------------------
    model_handle=dnn.BuildModel(weightsDir, INPUT_SIZE)

    # ---------------------   3、读图   ----------------------------
    for picDir in picDirList:
        image = cv2.imread(picDir)                     # 读图像,读进来直接是BGR 数据格式在0~255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 将图像转换为RGB
        T1 = time.time()
        frame, inferResult = dnn.inference_and_bbox(image, labels_list,model_handle,
                                                    INPUT_SIZE, SCORE_THRESHOLD,IOU_THRESHOLD)
        #inferResult= {'box_num': 2, 'Dry_garbage': 2,
        # 'object_list': [[805, 436, 495, 629, 'Dry_garbage'], [219, 434, 506, 587, 'Dry_garbage']]}
        T2 = time.time()
        folder=os.path.dirname(picDir)
        filename_contend=os.path.basename(picDir)

        # 生成xml文件
        if "object_list" in inferResult:
            #'{0}/{1}.xml'.format(folder,filename_contend.split(".")[0]
            creatXml.byDnn(filename_contend,folder, inferResult["object_list"])

        if show_handle:
            cv2.namedWindow("frame", 0)  # cv2.WINDOW_NORMAL就是0，窗体可以自由变换大小
            cv2.imshow("frame", frame)
            # 键盘监听，间隔1ms，监听到'q'后执行下面的break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # 跳出while循环体
                break


if __name__=="__main__":
    path = "config.json"
    configList = load_json_config(path)
    folderDir=configList[0]
    endswithList=configList[1]
    labelDir=configList[2]
    weightsDir=configList[3]
    show_handle=configList[4]

    # ---------------------   输入文件夹   --------------------------
    re=file_check_from_folder(folderDir, endswithList)
    is_supported, picDirList, not_right_file_list, not_right_extension_list=re
    assert is_supported, '文件夹不存在或者文件夹下不满足都是支持的文件！【20210710-01】'


    # ---------------------   模型相关参数   ------------------------
    # 输入模型的地址
    current_folder = os.path.dirname(os.path.abspath(__file__))
    weights_absoluteDir = os.path.join(current_folder, weightsDir)
    label_absoluteDir = os.path.join(current_folder, labelDir)
    # 输入模型时图片的尺寸
    INPUT_SIZE = 416
    # 是目标的概率
    SCORE_THRESHOLD = 0.6
    # 目标和背景的阈值
    IOU_THRESHOLD = 0.5


    # ---------------------   调用模型生成xml文件   --------------------------
    generateXml(picDirList,label_absoluteDir, weights_absoluteDir,
                INPUT_SIZE, SCORE_THRESHOLD,IOU_THRESHOLD,show_handle)
