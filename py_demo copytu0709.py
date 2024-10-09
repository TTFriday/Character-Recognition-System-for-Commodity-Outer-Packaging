import sys
import os
import ctypes
import numpy as np
import cv2
import time 
import threading
from PIL import Image
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont  
from pyzbar.pyzbar import decode, ZBarSymbol 

# 定义MatchResult结构体
class MatchResult(ctypes.Structure):
    _fields_ = [
        ('leftTopX', ctypes.c_double),
        ('leftTopY', ctypes.c_double),
        ('leftBottomX', ctypes.c_double),
        ('leftBottomY', ctypes.c_double),
        ('rightTopX', ctypes.c_double),
        ('rightTopY', ctypes.c_double),
        ('rightBottomX', ctypes.c_double),
        ('rightBottomY', ctypes.c_double),
        ('centerX', ctypes.c_double),
        ('centerY', ctypes.c_double),
        ('angle', ctypes.c_double),
        ('score', ctypes.c_double)
    ]

# 定义Matcher类
class Matcher:
    def __init__(self, dll_path, maxCount, scoreThreshold, iouThreshold, angle, minArea):
        self.lib = ctypes.CDLL(dll_path)
        self.lib.matcher.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.lib.matcher.restype = ctypes.c_void_p
        self.lib.setTemplate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.match.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(MatchResult), ctypes.c_int]        
        if maxCount <= 0:
            raise ValueError("maxCount must be greater than 0")
        self.maxCount = maxCount
        self.scoreThreshold = scoreThreshold
        self.iouThreshold = iouThreshold
        self.angle = angle
        self.minArea = minArea        
        self.matcher = self.lib.matcher(maxCount, scoreThreshold, iouThreshold, angle, minArea)
        self.results = (MatchResult * self.maxCount)()
    
    def set_template(self, image):
        height, width = image.shape[0], image.shape[1]
        channels = 1
        data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        return self.lib.setTemplate(self.matcher, data, width, height, channels)
    
    def match(self, image):
 
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                raise ValueError("Invalid image shape")
        else:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                raise ValueError("Invalid image shape")
        height, width = image.shape[0], image.shape[1]
        channels = 1
        data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        return self.lib.match(self.matcher, data, width, height, channels, self.results, self.maxCount)
    
def show_code(img):
    image = cv2.imread(img)
    try:
        # BARCodes = decode(image) 
        BARCodes = decode(image, symbols=[ZBarSymbol.EAN13])   
        for BARCode in BARCodes:            
            stringData = BARCode.data.decode('utf-8') 
            # print (stringData)  # stringData为条形码数据
        if not BARCodes :  
            print("未能检测出该图片的条形码。")
    except Exception as e:  
    # 处理异常，但注意这不会捕获警告  
        print(f"An exception occurred: {e}")

    return stringData
def show_Qrcode(img):
    image = cv2.imread(img)  
 
    # 解码图片中的二维码  
    try:
        decoded_objects = decode(image, symbols=[ZBarSymbol.QRCODE]) 
        # 遍历解码结果并打印  
        for obj in decoded_objects:  
            # 检查是否为二维码（QR Code）  
            
            stringData_qr = obj.data.decode('utf-8')    
            # print('二维码:', obj.data.decode("utf-8"))  # 解码数据为字符串  
        
        # 如果没有找到二维码  
        if not decoded_objects:  
            print("未能检测出该图片的二维码。")
    except Exception as e:  
    # 处理异常，但注意这不会捕获警告  
        print(f"An exception occurred: {e}")
    return stringData_qr
# ------------------ 上的是二维码和条形码的------------------- #

# ------------------ 下面是开始找模板------------------------ #

def list_angle(start_angle,end_angle):
    '''
    找角度
    '''
    lst = []
    for i in range(start_angle,end_angle,5):
        lst.append(i)
        lst_with_abs = [(abs(num), num) for num in lst]  
        lst_with_abs.sort(key=lambda x: x[0])  
        sorted_lst = [num for _, num in lst_with_abs]
    return sorted_lst

def out_image(start_angle,end_angle,maxCount,scoreThreshold, iouThreshold, minArea, image, tmp):    
    dll_path = './templatematching_ctype.dll' # 模板匹配库路径
    for angle in list_angle(start_angle,end_angle):  
        start = time.time()  
        matcher = Matcher(dll_path, maxCount, scoreThreshold, iouThreshold, angle, minArea)
        if matcher is None:
            print("Create Matcher failed")
            sys.exit(111)    
        if image is None:
            print("Read image failed")
            sys.exit(111)
        matcher.set_template(image)
        # cap1 = cv2.imread(tmp)
        cap = tmp.copy()
        mask = np.zeros(cap.shape[:2], dtype="uint8")  
        matches_count = matcher.match(cap)
        # print (matches_count)
        if matches_count < 0:
            print("Match failed!")
        if matches_count == 0 :
            continue
        if matches_count == 1:
            assert matches_count <= matcher.maxCount, "matches_count must be less than or equal to maxCount"
            for i in range(min(matches_count, matcher.maxCount)):
                result = matcher.results[i]
                if result.score > 0:
                    polygon  = np.array([[result.leftTopX, result.leftTopY], [result.leftBottomX, result.leftBottomY], [result.rightBottomX, result.rightBottomY], [result.rightTopX, result.rightTopY]], np.int32)
                    polygons = [np.array([[result.leftTopX, result.leftTopY], [result.leftBottomX, result.leftBottomY], [result.rightBottomX, result.rightBottomY], [result.rightTopX, result.rightTopY]], np.int32)] 
                    rect = cv2.minAreaRect(polygon)  
                    box = cv2.boxPoints(rect)  
                    box = np.intp(box)  
                    center_x,center_y,W,h =find_center(box)

                    
                    cv2.drawContours(cap, [box], 0, (0, 0, 255), 2)  # 红色矩形  
                    M = cv2.getRotationMatrix2D((center_x,center_y), -result.angle, 1.0)
                    # img_rot = cv2.warpAffine(cap, M, (cap.shape[1], cap.shape[0])) 
                    img_rot = cv2.warpAffine(cap, M, (cap.shape[1], cap.shape[0]))  
                    img_crop = cv2.getRectSubPix(img_rot, (W,h), (center_x,center_y))
                    cv2.polylines(cap, polygons , True, (0, 255, 0), 1)
                    cv2.putText(cap, str(result.score), (int(result.leftTopX), int(result.leftTopY)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            break
    return cap, img_crop

def find_center (points):
    x_min, y_min = min(p[0] for p in points), min(p[1] for p in points)  
    x_max, y_max = max(p[0] for p in points), max(p[1] for p in points)  
    W = x_max - x_min  
    h = y_max - y_min   
    center_x = (x_min + x_max) / 2  
    center_y = (y_min + y_max) / 2  
    return center_x,center_y,W,h

def outputlist (img):
    kernel=np.ones((5,5),np.uint8)
    kernel2=np.ones((8,8),np.uint8)    
    erosion=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    output_list = [] 
    ocr = PaddleOCR(use_angle_cls=True, lang="en")   
    img_rgb = cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB)  
    result = ocr.ocr(img_rgb)  
    for line in result: 
        for i in range(len(line)):
            output_list.append(line[i][1][0])    
    return output_list

def write_img (img,output_list):
    # img = cv2.imread(img)
    image1 = cv2.resize(img,(640,640))
    window_name1 = 'image'
    font1 = cv2.FONT_HERSHEY_SIMPLEX
    org1 = (10, 50)
    fontScale1 = 1
    color1 = (0, 255, 255)
    thickness1 = 2
    for line in output_list:
        image_1 = cv2.putText(image1, line, org1, font1, fontScale1, color1, thickness1, cv2.LINE_AA)
        x,y = org1
        y = y+30
        org1 = (x,y)
    return image_1

if __name__ == '__main__':   
    maxCount = 1
    scoreThreshold = 0.4
    iouThreshold = 0.3
    start_angle = -30
    end_angle = 60
    minArea = 256
    image = cv2.imread('.//testimg//testa//model//b1.jpg', cv2.IMREAD_GRAYSCALE)
    
    dir_origin_path = ".//testimg//testa//b1_img//"
    save_path = './/testimg//save//'

    img_names = os.listdir(dir_origin_path)  
    for img_name in img_names:
        image_path = os.path.join(dir_origin_path, img_name) 
        print (image_path)
        # dect_ocr  = macth_ocr()
        tmp = cv2.imread(image_path)
        # cv2.imshow("ls",tmp)
        # cv2.waitKey(0)
        c_img,r_img = out_image(start_angle,end_angle,maxCount,scoreThreshold, iouThreshold, minArea,image,tmp)     
        outlist = outputlist(r_img)    
        r_img = write_img(r_img,outlist)
        # str_len13 = match_ocr.show_code(tmp)
        # str_qt = match_ocr.show_Qrcode(tmp)            
    

        
        save_path_file = os.path.join(save_path, img_name) 
        print (save_path_file)
        cv2.imwrite(save_path_file,r_img)
        cv2.waitKey(0)

    # image = cv2.imread('.//testimg//testa//model//c.jpg', cv2.IMREAD_GRAYSCALE)
    # tmp = ".//testimg//testa//c_img//image01.jpg"
    # cap,r_img = out_image(start_angle,end_angle,maxCount,scoreThreshold, iouThreshold, minArea,image,tmp)
    # (h,w,_)=r_img.shape    
    # outlist = outputlist(r_img)
    # r_img = write_img(r_img,outlist)
    # # print (show_code(tmp))
    # # print (show_Qrcode(tmp))
    # # cv2.imshow("test",r_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()




