# coding:utf-8
# jn10010537
# config.py
import sys
import json
# ------  读取配置文件   --------------
def load_json_config(path):
    try:
        # 注意有中文，所以编码要注明是'utf-8'
        with open(path, 'r', encoding='utf-8') as f:
            # 函数是将json格式数据转换为字典
            define_json = json.load(f)
        folderDir=define_json["folderDir"]
        endswithList=define_json["endswithList"]
        labelDir = define_json["labelDir"]
        weightsDir=define_json["weightsDir"]
        show_handle = define_json["show_handle"]
        if show_handle:
            print("\n>>>读取参数：")
            print(">>>文件夹的路径：",folderDir)
            print(">>>支持的文件后缀列表：",endswithList)
            print(">>>标签路径：", labelDir)
            print(">>>权重文件的路径：", weightsDir)
            print(">>>是否显示以及打印后台信息：", show_handle)
    except Exception as e:
        print(">>>配置文件读取失败！")
        print(e)
        print(">>>程序即将关闭！")
        sys.exit()

    return folderDir,endswithList,labelDir,weightsDir,show_handle

if  __name__=="__main__":
    path="config.json"
    configList=load_json_config(path)
