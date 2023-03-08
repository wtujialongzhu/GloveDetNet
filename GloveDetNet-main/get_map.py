import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from GloveDET import CAPN
if __name__ == "__main__":
    '''
    Recall and Precision are not like APs as an area of concept, so when the confidence is different, the recalted and precision values of the network are different.。
    By default, the RECALL and Precision calculated in this code represent the corresponding RECALL and Precision values when the confidence is 0.5.
    Limited by the principle of MAP calculation, the network needs to obtain nearly all prediction boxes when calculating MAP, so as to calculate the recall and precision values under different limits conditions
    Therefore, the number of txts in the map_out/detection-results/in this code is generally more in the number of TXT boxes in the box.，
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode is used to specify the content calculated when the file is running
    #   MAP_MODE is 0 for the entire MAP computing process, including obtaining prediction results, obtaining real boxes, and calculating VOC_MAP.
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    #   Classes_path here is used to specify a category that needs to measure VOC_MAP
    #   Under normal circumstances, it is consistent with the CLASSSSES_PATH used in training and prediction
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/coco_classes.txt'
    MINOVERLAP      = 0.5
    confidence      = 0.001
    nms_iou         = 0.5
    score_threhold  = 0.5
    map_vis         = False
    VOCdevkit_path  = 'VOCdevkit'
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = CAPN(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
