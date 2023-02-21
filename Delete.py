import os
path = 'huancun'  # 文件路径
if os.path.exists(path):  # 如果文件存在
    print("clear huancun")
    datanames = os.listdir(path)
    list = []
    for i in datanames:
        list.append(i)
        mydir = path + '/' + i
        print(mydir)
        try:
            os.remove(mydir)
            print("delete success")
        except Exception as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


else:
    print('no such file:%s' % path)  # 则返回文件不存在