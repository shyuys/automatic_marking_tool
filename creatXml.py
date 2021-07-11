# -*- coding: utf-8 -*-
# author:jn10010537
# 本脚本byDnn(filename_contend,folder, *args, **kw)函数创建xml;
# 本脚本的输入是考虑opencv-dnn接口，["xmin", "ymin", 'h', 'w', 'label']作为byDnn的输入；

import os
import sys
import logging
from xml.dom.minidom import Document

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

def filterParameter(tuple_args,dict_kwargs):
    '''
    用于过滤参数,函数遵循单入口，单出口；
    如果有返回值，则返回[xmin,ymin,xmax,ymax,label]；否则返回None;

    :param tuple_args:  元组参数；
    :param dict_kwargs: 字典参数；
    :return:None 或者 [[xmin, ymin, xmax, ymax, label],...]
    '''
    print("\n\n过滤byDnn(a,b,*args,**kw)中不定长*args,**kw参数... ... ")
    re=None
    if bool(tuple_args) or bool(dict_kwargs):

        # --------- 优先解析tuple_args参数 -----------------------------------------------------------------------------
        if bool(tuple_args):
            log.info("解析元组数据... ... ")

            if type(tuple_args[0]) == type(1) and (5 == len(tuple_args)):
                log.info("解析字元组据格式是int数字... ... ")
                temp_list = []
                xmin = tuple_args[0]
                ymin = tuple_args[1]
                xmax = xmin + tuple_args[2]
                ymax = ymin + tuple_args[3]
                label = tuple_args[4]
                temp_list.append([xmin, ymin, xmax, ymax, label])
                re = temp_list

            elif type(tuple_args[0]) == type(list()):
                log.info("解析字典数据格式是列表... ... ")
                list_temp = list(set(len(i) for i in tuple_args[0]))
                if 1 == len(list_temp) and 5==list_temp[0]:
                    temp_list = []
                    for args in tuple_args[0]:
                        xmin = args[0]
                        ymin = args[1]
                        xmax = xmin + args[2]
                        ymax = ymin + args[3]
                        label =args[4]
                        temp_list.append([xmin, ymin, xmax, ymax, label])
                    re = temp_list
                else:
                    log.error(r"*args参数,列表的数据数量不等于5！错误索引码20210604-03")

        # ---------  解析dict_kwargs参数 -------------------------------------------------------------------------------
        else:
            key_list = ["xmin", "ymin", 'h', 'w', 'label']
            if bool(dict_kwargs) and (5 == len(dict_kwargs)) and \
                    (set(key_list) == set(dict_kwargs.keys())):
                log.info("解析字典数据... ... ")

                if type(dict_kwargs["xmin"])==type(1):
                    log.info("解析字典数据格式是int数字... ... ")
                    temp_list = []
                    xmin = dict_kwargs["xmin"]
                    ymin = dict_kwargs["ymin"]
                    xmax = xmin + dict_kwargs["w"]
                    ymax = ymin + dict_kwargs["h"]
                    label = dict_kwargs["label"]
                    temp_list.append([xmin, ymin, xmax, ymax, label])
                    re =temp_list

                elif type(dict_kwargs["xmin"]) == type(list()):
                    log.info("解析字典数据格式是列表... ... ")

                    list_temp = list(dict_kwargs.values())
                    list_temp = list(set(len(i) for i in list_temp))

                    if 1 == len(list_temp):
                        temp_list = []
                        for i in range(list_temp[0]):
                            xmin = dict_kwargs["xmin"][i]
                            ymin = dict_kwargs["ymin"][i]
                            xmax = xmin + dict_kwargs["w"][i]
                            ymax = ymin + dict_kwargs["h"][i]
                            label = dict_kwargs["label"][i]
                            temp_list.append([xmin, ymin, xmax, ymax, label])
                        re = temp_list
                    else:
                        log.error(r"**kw参数,列表的数据数量不一致！错误索引码20210604-02")
                else:
                    log.error(r"**kw参数为非int数字,非列表的数据格式，不受支持！错误索引码20210604-01")
            else:
                log.error(r"请正确输入(xmin,ymin,w,h,label),错误索引码20210602-02")
    else:
        log.error(r"请正确输入(xmin,ymin,w,h,label),错误索引码20210602-01")
    return re

def xml_write_object(doc,root,contents):
    '''
    向xml文件中写入<object>标签，内容如下：
    <object>
        <name>smoke</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>339</xmin>
            <ymin>3</ymin>
            <xmax>500</xmax>
            <ymax>166</ymax>
        </bndbox>
    </object>

    :param doc: 文档对象doc,创建文档对象doc=Document()
    :param root: 根节点；在文档里创建一个根节点root=doc.createElement('annotation')
    :param contents: 内容；形如：[[xmin, ymin, xmax, ymax, label],...]
    :return: None
    '''
    # 创建二级节点<object>
    object = doc.createElement("object")
    root.appendChild(object)

    # 创建三级节点<name>、<pose>、<truncated>、<difficult>、<bndbox>
    # 创建三级节点<name>
    name = doc.createElement("name")
    name.appendChild(doc.createTextNode(contents[4]))  # 添加文本节点
    object.appendChild(name)

    # 创建三级节点<pose>
    pose = doc.createElement('pose')
    pose.appendChild(doc.createTextNode('Unspecified'))  # 添加文本节点
    object.appendChild(pose)

    # 创建三级节点<truncated>
    truncated = doc.createElement('truncated')
    truncated.appendChild(doc.createTextNode('0'))  # 添加文本节点
    object.appendChild(truncated)

    # 创建三级节点<difficult>
    difficult = doc.createElement('difficult')
    difficult.appendChild(doc.createTextNode('0'))  # 添加文本节点
    object.appendChild(difficult)

    # 创建三级节点<bndbox>
    bndbox = doc.createElement('bndbox')
    object.appendChild(bndbox)

    # 创建四级节点<xmin>
    xmin = doc.createElement('xmin')
    xmin.appendChild(doc.createTextNode(str(contents[0])))  # 添加文本节点
    bndbox.appendChild(xmin)

    # 创建四级节点<ymin>
    ymin = doc.createElement('ymin')
    ymin.appendChild(doc.createTextNode(str(contents[1])))  # 添加文本节点
    bndbox.appendChild(ymin)

    # 创建四级节点<ymin>
    xmax = doc.createElement('xmax')
    xmax.appendChild(doc.createTextNode(str(contents[2])))  # 添加文本节点
    bndbox.appendChild(xmax)

    # 创建四级节点<ymin>
    ymax = doc.createElement('ymax')
    ymax.appendChild(doc.createTextNode(str(contents[3])))  # 添加文本节点
    bndbox.appendChild(ymax)

def byDnn(filename_contend,folder, *args, **kw):
    '''
    *args
    不定长参数（可变参数）,（这些可变参数在函数调用时自动组装为一个tuple）
    *参数收集所有未匹配的"位置参数"组成一个tuple对象;如果是*args，局部变量args指向此tuple对象；
    假如必要参数是x，传递参数为byDnn(x,xmin,ymin,w,h,label)

    **kw
    这也是不定长参数（可变参数）,（这些关键字参数在函数内部自动组装为一个dict）
    **参数收集所有未匹配的关键字参数组成一个dict对象；如果是*kw，局部变量kw指向此dict对象；
    假如必要参数是x，传递参数为byDnn(x,xmin=1,ymin=2,w=3,h=4,label="yys")；

    通过opencv的dnn返回值创建xml;dnn返回值可以通过*args，或者**kw传参；

    :param filename_contend:指定xml标签中的<filename>标签内容；
    :param folder:间接指定xml标签中的<path>标签内容；
    :param args:不定长参数，args指向tuple对象；
    :param kw:不定长参数，kw指向dict对象；
    :return:None
    '''

    # 创建一个文档对象doc
    doc=Document()

    # 在文档对象里，创建一个根节点
    root=doc.createElement('annotation')

    # 根节点加入到tree
    doc.appendChild(root)

    # 创建二级节点<filename>、<path>
    filename=doc.createElement('filename')
    filename.appendChild(doc.createTextNode(filename_contend)) # 添加文本节点
    root.appendChild(filename)

    path=doc.createElement('path')
    path.appendChild(doc.createTextNode(os.path.join(folder,filename_contend))) # 添加文本节点
    root.appendChild(path)

    contents=filterParameter(args,kw)

    # 有目标打框的对象
    if contents:
        for content in contents:
            xml_write_object(doc, root, content)

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open('{0}/{1}.xml'.format(folder,filename_contend.split(".")[0]),'w',encoding='utf-8') as fp:
        doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding='utf-8')

if __name__=="__main__":
    test_handle=3

    filename_contend="jn10010537.jpg"
    folder="./img"

    if 1.1==test_handle:
        # 测试 *args参数输入方式；
        # 只有一个打标对象；
        # 输入列表的元素分别为xmin, ymin, w, h, label
        # xmin, ymin, w, h, label分别为左上角点x,y，宽，高，标签；
        byDnn(filename_contend, folder,341,12,162,155,"smoke")

    elif 1.2 == test_handle:
        # 测试 *args参数输入方式；
        # 有多个打标对象；
        # 输入列表的元素分别为xmin, ymin, w, h, label
        # xmin, ymin, w, h, label分别为左上角点x,y，宽，高，标签；
        object_list1 = [341, 12, 162, 155, "smoke"]
        object_list2 = [541, 52, 962, 355, "fire"]

        byDnn(filename_contend, folder, [object_list1, object_list2])

    elif 2.1==test_handle:
        # 测试 **kw参数输入方式;
        # 只有一个打标对象；
        # xmin, ymin, w, h, label分别为左上角点x,y，宽，高，标签；
        byDnn(filename_contend, folder, xmin=341, ymin=12,w=162,h=155,label="smoke")

    elif 2.2==test_handle:
        # 测试 **kw参数输入方式;
        # 有多个打标对象；
        # xmin, ymin, w, h, label分别为左上角点x,y，宽，高，标签；
        byDnn(filename_contend, folder, xmin=[341,541], ymin=[12,52], w=[162,962], h=[155,355], label=["smoke","fire"])
    else:
        print(">>>请输入1.1、1.2、2.1、2.2演示生成xml~")



