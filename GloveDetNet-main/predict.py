#-----------------------------------------------------------------------#
#   predict.py    Single picture prediction, camera detection, FPS test and directory traversal detection and other functions
#-----------------------------------------------------------------------#
import time
import socket
import cv2
import numpy as np
from PIL import Image
from SendPic import runserver
from GloveDET import CAPN
import os
import threading
import _thread
global iszhengchang
iszhengchang=1
if __name__ == "__main__":
    detect = CAPN()
    #----------------------------------------------------------------------------------------------------------#
    #   MODE is used to specify the test mode：
    #   'predict'           Indicates a single picture prediction. If you want to modify the prediction process, such as saving pictures, intercepting objects, etc., you can first look at the detailed annotation below
    #   'video'             Express video detection, you can call the camera or video for detection, and check the comment below for details.
    #----------------------------------------------------------------------------------------------------------#
    mode = "video"
    #-------------------------------------------------------------------------#
    #   crop                Specify whether the target is intercepted after the single picture prediction
    #   count               Specify whether the target counts
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          The path for specifying the video, when Video_Path = 0 indicates the detection camera
    #                       If you want to detect the video, set it like video_path = "xxx.mp4", which means that you read the xxx.mp4 file in the root directory.
    #   video_save_path     Indicates the path of video preservation, indicates not saved when Video_save_path = ""
    #                       If you want to save the video, set it like video_save_path = "yyy.mp4", which represents the yyy.mp4 file that is saved as the root directory.
    #   video_fps           FPS for video for saved
    #
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     Specify the folder path for the picture for detection
    #   dir_save_path       Specify the preservation path of the detection picture
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    heatmap_save_path = "model_data/heatmap_vision.png"
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = detect.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        num=1
        time1=time.time()
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failure to read the camera correctly (video), please pay attention to whether to install the camera correctly (whether to fill in the video path correctly).")
        fps = 0.0
        while(True):
            t1 = time.time()
            b = time.localtime(t1)
            c = time.strftime("%Y-%m-%d %H:%M:%S", b)
            c = c.split(' ')
            c[1] = c[1].replace(':', 'H', 1)
            c[1] = c[1].replace(':', 'M')
            c[1] = c[1] + 'S'
            c[0] = c[0].replace('-', '_')
            name = str(c[0]) + "_" + str(c[1])
            # Read a certain frame
            ref, frame = capture.read()


            if t1-time1 > 4:
                # cv2.imwrite(file+'/'+str(num) +".jpg", frame)
                cv2.imwrite('huancun/'+str(c[0]) + "_" + str(c[1]) + ".jpg", frame)
                # print(file+'/'+str(num) +".jpg")
                num = num + 1
                time1=t1
                try:
                    if iszhengchang == 0:
                        runserver('huancun/' + str(c[0]) + "_" + str(c[1]) + ".jpg")
                except:
                    print("try to reconnect")

            if not ref:
                break
            # Format change，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Transform into images
            frame = Image.fromarray(np.uint8(frame))
            # Test
            frame = np.array(detect.detect_image(frame))
            # RGBTOBGR satisfies the OpenCV display format
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = detect.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = detect.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                detect.detect_heatmap(image, heatmap_save_path)
                
    elif mode == "export_onnx":
        detect.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
