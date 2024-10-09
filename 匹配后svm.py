from paddleocr import PaddleOCR, draw_ocr
import cv2
import os
import random
import numpy as np
import pickle
from PIL import Image
import time
from class_py_demo import macth_ocr

'''
轮廓匹配分割字符排序-->svm识别输出结果
'''
class Rectangle:  
    def __init__(self, w, h):  
        self.w = w  
        self.h = h  
        self.arce = w * h 

class Rectangle_box:  

    def __init__(self, x1, y1, x2, y2):  
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2  
  
    def intersects(self, other):  
        # 检查两个矩形是否在水平或垂直方向上有重叠  
        return not (self.x2 <= other.x1 or self.x1 >= other.x2 or self.y2 <= other.y1 or self.y1 >= other.y2)  
  
    def merge(self, other):  
        # 合并两个矩形  
        return Rectangle_box(  
            min(self.x1, other.x1),  
            min(self.y1, other.y1),  
            max(self.x2, other.x2),  
            max(self.y2, other.y2)  
        )  

class public_img(): 
    def __init__(self): 
        self.svm_ocr = SVM_ocr()

    def points_to_xywh(self,point1, point2):  
        '''
        (x_min,y_min)(x_max,y_max) (xyxy)转（xywh） 
        '''
        x1, y1 = point1  
        x2, y2 = point2    
        if x1 > x2:  
            x1, x2 = x2, x1  
        if y1 > y2:  
            y1, y2 = y2, y1  
        width = x2 - x1  
        height = y2 - y1   
        return [int(x1), int(y1), int(width), int(height)]  
    
    def points_to_xyxy(self,x,y,w,h):  
        '''
        (x_min,y_min)(x_max,y_max) (xywh)转（xyxy） 
        '''
        x1, y1 = x,y 
        x2, y2 = x+w,y+h  
        if x1 > x2:  
            x1, x2 = x2, x1  
        if y1 > y2:  
            y1, y2 = y2, y1  
        return [int(x1), int(y1), int(x2), int(y2)]  
    
    def boxes_and_img(boxs, img, output_dir,tem_list):    
        """    
        boxs : [(label, x, y, w, h), ...]    
        img : 图像    
        output_dir : 保存图像的目录    
        返回 ：在img 上画框（但不返回），并截图保存到output_dir    
        """    
        if not os.path.exists(output_dir):    
            os.makedirs(output_dir)    
        i = 0
        for box in boxs:    
            label, x, y, w, h = box    
            x, y, w, h = int(x), int(y), int(w), int(h)  
            x_min, y_min = max(0, x), max(0, y)  
            x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)  
            cropped_img = img[y_min:y_max, x_min:x_max] 
            if label in tem_list: 
                label = label +"//"
                label_dir = os.path.join(output_dir,label)    
                if not os.path.exists(label_dir):    
                    os.makedirs(label_dir)     
                file_name = f"image_{random.randint(1000000, 9999999)}_{i}.jpg"  # 使用简单的文件名  
                i = i+1
                file_path = os.path.join(label_dir, file_name)  
                # print (file_path)  
                cv2.imwrite(file_path, cropped_img)  
        
        # 如果需要，返回修改后的图像  
        return img
    
    def boxes_and_img_save(self,img, output_dir,list,tem_list):    
        """    
        boxs : [ (x, y, w, h,label), ...]    
        img : 图像    
        output_dir : 保存图像的目录    
        返回 ：在img 上画框（但不返回），并截图保存到output_dir    
        """    
        if not os.path.exists(output_dir):    
            os.makedirs(output_dir)    
        i = 0
        for box in list:    
            x, y, w, h, label= box    
            x, y, w, h = int(x), int(y), int(w), int(h)  
            x_min, y_min = max(0, x), max(0, y)  
            x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)  
            cropped_img = img[y_min:y_max, x_min:x_max]     

    def boxes_and_img_save(self,img, output_dir,list,tem_list):    
        """    
        boxs : [ (x, y, w, h,label), ...]    
        img : 图像    
        output_dir : 保存图像的目录    
        返回 ：在img 上画框（但不返回），并截图保存到output_dir    
        """    
        if not os.path.exists(output_dir):    
            os.makedirs(output_dir)    
        i = 0
        for box in list:    
            x, y, w, h, label= box    
            x, y, w, h = int(x), int(y), int(w), int(h)  
            x_min, y_min = max(0, x), max(0, y)  
            x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)  
            cropped_img = img[y_min:y_max, x_min:x_max] 

            if label in tem_list:
                if label =="年":
                    label = "ch_nian"
                if label =="月":
                    label = "ch_yue"
                if label =="日":
                    label = "ch_ri"    
                label = label +"//"                
                label_dir = os.path.join(output_dir,label)    
                if not os.path.exists(label_dir):    
                    os.makedirs(label_dir)                         
                file_name = f"image_{random.randint(1000000, 9999999)}_{i}.jpg"  # 使用简单的文件名  
                i = i+1
                file_path = os.path.join(label_dir, file_name)  
                # print (file_path)  
                cv2.imwrite(file_path, cropped_img)          
        # 如果需要，返回修改后的图像  
        # return img
    def img_box_lk(self,img,max_arce,min_arce):
        '''
        通过轮廓画框，加上判断函数
        return  画框图像，轮廓坐标
        '''
        kxy = []
        kwh = []
        kxywh = []
        img = self.eroded_and_dilated_img(img)
        contours, _ = cv2.findContours(img,cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)  
        for contour in contours:  
            x, y, w, h = cv2.boundingRect(contour) 
            arce = w*h 
            if max_arce> arce >= min_arce:
                kxy.append([x,y])
                kwh.append(Rectangle(w,h))  
                kxywh.append([x,y,w,h])
                x, y, w, h = int(x), int(y), int(w), int(h)  
                x_min, y_min = max(0, x), max(0, y)  
                x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)  
                # cv2.rectangle(img, (x_min, y_min), (x_max , y_max), (255, 255, 0), 2)
        return kxywh
    
    def img_box_lk_ocr(self,img,max_arce,min_arce,txt):
        '''
        和上面的img_box_lk 作用相同加入排须
        通过轮廓画框，加上判断函数
        return  画框图像，轮廓坐标
        '''
        # kxy = []
        # kwh = []
        kxywh = []
        k_ls = []
        k_ls1 = []
        img = self.eroded_and_dilated_img(img)
        contours, _ = cv2.findContours(img,cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)  
        for contour in contours:  
            x, y, w, h = cv2.boundingRect(contour) 
            arce = w*h 
            if max_arce> arce >= min_arce:
                # kxy.append([x,y])
                # kwh.append(Rectangle(w,h))  
                
                x, y, w, h = int(x), int(y), int(w), int(h)  
                kxywh.append([x,y,w,h])
                x_min, y_min = max(0, x), max(0, y)  
                x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h) 
        kxl = self.find_and_sort_on_same_line(kxywh)
        
        inner_list=[]    
        # print(kxl)
        if len(kxl) > 0: 
            inner_list = kxl[0]  
    
            # 遍历内部列表和data_str的字符  
            for i, sublist in enumerate(inner_list):  
                # 在每个子列表中添加对应的字符（假设data_str足够长）  
                sublist.append(txt[i]) 
        return inner_list
    
    def find_and_sort_on_same_line(self,lst):  
        '''
        lst [[],[]]
        返回 ls [[[],[]],[[],[]]]
        '''
        ls = []
        while lst:             
            min_xy_sublist = min(lst, key=lambda x: x[0] + x[1])   
            reference_y = min_xy_sublist[1]  
            reference_y_plus_h = reference_y + min_xy_sublist[3]   
            same_line_sublists = [sublist for sublist in lst if reference_y-10 <= sublist[1] < reference_y_plus_h]             
            if same_line_sublists:  
                list2 = sorted(same_line_sublists, key=lambda x: x[0])  
                ls.append(list2)
                for sublist in same_line_sublists:  
                    lst.remove(sublist)  
            else:  
                ls.append(min_xy_sublist) 
                lst.remove(min_xy_sublist)  
        return ls 
    
    def list_to_boxes_in_img (self,img1,list):
        '''
        已经分行
        根据不同的list 在图片上画框 [[[]]]
        label 的信息以使用
        '''
        img = img1.copy()
        # print (list)
        for listi in list:
            for box in listi: 
                if len(box)==5:     
                    _, x, y, w, h = box    
                    x, y, w, h = int(x), int(y), int(w), int(h)  
                    x_min, y_min = max(0, x), max(0, y)  
                    x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)  
                    cv2.rectangle(img, (x,y), (x+w , y+h), (0, 255, 0), 2)
                if len(box)==4:   
                # label, x, y, w, h = box    
                    x, y, w, h = box    
                    x, y, w, h = int(x), int(y), int(w), int(h)  
                    x_min, y_min = max(0, x), max(0, y)  
                    x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)  
                    cv2.rectangle(img, (x_min, y_min), (x_max , y_max), (0, 255, 0), 2)
        return img
    
    def listone_to_boxes_in_img (self,img1,list):
        '''
        未分行
        根据不同的list 在图片上画框 [[[]]]
        label 的信息以使用
        '''
        img = img1.copy()
        for box in list: 
            # print (box)
            if len(box)==5:   
            # label, x, y, w, h = box    
                _, x, y, w, h = box    
                x, y, w, h = int(x), int(y), int(w), int(h)  
                x_min, y_min = max(0, x), max(0, y)  
                x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)  
                cv2.rectangle(img, (x,y), (x+w , y+h), (0, 255, 0), 2)
            if len(box)==4:   
            # x, y, w, h = box    
                x, y, w, h = box    
                x, y, w, h = int(x), int(y), int(w), int(h)  
                x_min, y_min = max(0, x), max(0, y)  
                x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)  
                cv2.rectangle(img, (x_min, y_min), (x_max , y_max), (0, 255, 0), 2)
        return img
    
    def one_find(self,boxes, hang): 
        one_list= []
        two_list= []
        three_list = [] 
        hang = int(hang)
        for box in boxes:  
            _, _, _, y1,y2 = box       
            center_y = int((y1 + y2) // 2)
            if center_y in range(0,hang):
                one_list.append(box)
            if center_y in range(hang,hang*2):
                two_list.append(box)
            if center_y in range(hang*2,hang*3):
                three_list.append(box)
        one = self.sortlisttest(one_list)
        two = self.sortlisttest(two_list)
        three = self.sortlisttest(three_list)
        return one,two,three
 
    def sortlisttest (list):
        list_sort= []
        sorted_boxes = sorted(list, key=lambda box: (box[1] + box[2]) // 2)     
        for box in sorted_boxes:  
            list_sort.append(box[0])
        return list_sort


    def eroded_and_dilated_img(self,img):
        '''
        侵蚀图像
        img ： 图像 输出： 侵蚀后的图像
        '''
        # cv2.imshow("test",img)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # 需要调试
        _, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)  
        # 腐蚀 ---> 白色前 ，黑色背景
        eroded = cv2.erode(thresh, kernel_1)
        dilated = cv2.dilate(eroded, kernel)
        return thresh
    
    def ocr_img_clean(self,img_path,img):
        
        '''
        ocr 识别提供类别label  
        有变动 注意
        '''
        ls_k = []
        ocr = PaddleOCR(use_angle_cls=True, lang="ch")
        result = ocr.ocr(img_path, cls=True)
        for idx in result:            
            for line in idx:
                txt = line[1][0]
                boxes_x1y1 = line[0][0]
                boxes_x2y2 = line[0][2]
                x1 , y1  = boxes_x1y1
                x2, y2  = boxes_x2y2
                
                ls_k.append([txt,x1,y1,x2,y2])          
        return ls_k

                
    def line_yy_and_clean(self,list1 ,list2):
       '''
       判断轮廓框（list1）的中心点是否在ocr识别的连续框（list2）内
       两个list 去匹配 
       return list  带有label x,y,w,h
       '''
       print (len(list1))
       list = [[],[],[],[],[],[]]
       list_2 = [[],[],[],[],[],[]]
       for i in range(len(list2)):
            txt,x1,y1,w1,h1 = list2[i]
            for box in list1:
                x,y,w,h = box
                center_x1 = x+w/2
                center_y1 = y+h/2
                if (center_x1 >= x1 and center_x1 <=x1+w1) and (center_y1 >= y1 and center_y1 <= y1+h1): 
                    list[i].append(box)
            sorted_data = sorted(list[i], key=lambda item: item[0]) 
            if len(sorted_data) <= len(txt):
                for ls_i in range(len(list[i])):
                    x,y,w,h = sorted_data[int(ls_i)]
                    label = txt[ls_i]
                    list_2[i].append([label,x,y,w,h]) 
            else:
                for ls_i in range(len(txt)):
                    x,y,w,h = sorted_data[int(ls_i)]
                    label = txt[ls_i]
                    list_2[i].append([label,x,y,w,h]) 
       return list_2

class SVM_ocr ():  
    def deskew(self,img):
        """去偏斜处理"""
        m = cv2.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * img.shape[1] * skew], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return img

    def preprocess_hog(self,images, max_len):
        """计算图像HOG描述子"""
        winSize = (20, 20)
        blockSize = (10, 10)
        blockStride = (5, 5)
        cellSize = (10, 10)
        nbins = 9
        derivAperture = 1
        winSigma = -1
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradient = True

        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                                derivAperture, winSigma, histogramNormType, L2HysThreshold,
                                gammaCorrection, nlevels, signedGradient)
        hog_descriptors = [hog.compute(img).flatten() for img in images]
        hog_descriptors = np.array([np.pad(desc, (0, max_len - len(desc)), 'constant') for desc in hog_descriptors], dtype=np.float32)
        return hog_descriptors

    def preprocess_image(self,image_path, size=(28, 28)):
        """规范化处理成固定尺寸的二值图像"""
        with Image.open(image_path) as img:
            img = img.convert('L')
            img_array = np.array(img)
            mean_pixel = np.mean(img_array)
            # 判断背景颜色
            if mean_pixel < 128:
                binary_array = np.where(img_array > 128, 0, 255).astype(np.uint8)
            else:
                binary_array = np.where(img_array < 128, 0, 255).astype(np.uint8)
            coords = np.column_stack(np.where(binary_array < 255))
            if coords.size == 0:
                raise ValueError(f"Image at path {image_path} is entirely white.")
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            cropped_img = img.crop((x_min, y_min, x_max + 1, y_max + 1))
            cropped_img.thumbnail(size, Image.LANCZOS)
            background = Image.new('L', size, 255)
            paste_position = ((size[0] - cropped_img.size[0]) // 2, (size[1] - cropped_img.size[1]) // 2)
            background.paste(cropped_img, paste_position)

            img_array = np.array(background)
            mean_pixel = np.mean(img_array)
            img_array = np.where(img_array < mean_pixel, 255, 0).astype(np.uint8)
            return img_array

    def load_model(self,model_path):
        """加载模型"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def predict(self,image_path, model, deskew_func, hog_func, max_len):
        """推理"""
        # 开始计时
        img_array = self.preprocess_image(image_path)
        img = deskew_func(img_array)
        img_hog = hog_func([img], max_len)
        start_time = time.time()
        prediction = model.predict(img_hog)
        end_time = time.time()  # 结束计时
        print(f"Processing time: {end_time - start_time} seconds")
        return prediction[0]

if __name__ == "__main__":
    min_arce = 3000
    max_arce = 40000
    start_angle = -60
    end_angle = 60
    
    maxCount = 1
    scoreThreshold = 0.5
    iouThreshold = 0.4
    minArea = 256
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    template_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','年','月','日','L','A','K','B','f']
    dll_path = './templatematching_ctype.dll'
    # template_folder_path = './/testimg//yao//data//' 
    image = cv2.imread('.//testimg//testa//model//b.jpg', cv2.IMREAD_GRAYSCALE)
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
    dir_origin_path = ".//testimg//testa//b_img//"
    output_dir = ".//zi//"
    dect_ocr  = macth_ocr()
    gy = public_img()
    svm_ocr = SVM_ocr()
    letters_digits_model_path = "svm_letters_digits.dat"
    # 加载模型和最大HOG描述符长度
    letters_digits_model, max_len = svm_ocr.load_model(letters_digits_model_path)
    # 图像路径
    img_path_letters_digits = 'test_image.png'
    # 预测
    result_letters_digits = svm_ocr.predict(img_path_letters_digits, letters_digits_model, deskew, preprocess_hog, max_len)
    print(f"Predicted label for letters/digits: {result_letters_digits}")
    img_names = os.listdir(dir_origin_path)  
    image_1 = ".//testimg//testa//b_img//image007.jpg" 
    image_1 = cv2.imread(image_1)
    img_path = "reslute.jpg"
    c_img,rn_img = dect_ocr.out_image(start_angle,end_angle,maxCount,scoreThreshold, iouThreshold, minArea,image,image_1)                 
    