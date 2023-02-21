import socket
import os
import sys
import struct
def socket_client(filepath):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 修改ip
        s.connect(('10.176.133.221', 5556))  # 此处ip必须为服务器端的ip

    except socket.error as msg:
        print(msg)
        sys.exit(1)

    while True:
        if os.path.isfile(filepath):
            # 定义定义文件信息。128s表示文件名为128bytes长，l表示一个int或log文件类型，在此为文件大小
            fileinfo_size = struct.calcsize('128sq')
            # 定义文件头信息，包含文件名和文件大小
            fhead = struct.pack('128sq', bytes(os.path.basename(filepath).encode('utf-8')),
                                os.stat(filepath).st_size)
            s.send(fhead)
            print('client filepath: {0}'.format(filepath))
            with open(filepath, 'rb') as fp:
                while True:
                    data = fp.read(1024)
                    if not data:
                        print('{0} file send over...'.format(filepath))
                        break
                    s.send(data)
        s.close()
        break
def runserver(filepath):
    # filepath = '12.html'
    if os.path.exists(filepath):
        socket_client(filepath)
    else:
        print("no such file ")

# socket_client("huancun/1.jpg")
