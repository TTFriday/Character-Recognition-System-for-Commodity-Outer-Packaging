# by yu 0611
# 应付多数的存在
 
# 完成 0611
# 针对b_img 加svm 
#  svm 调用正常结果混乱

import cv2
import os
import random
from class_py_demo import macth_ocr
import os
import cv2
import numpy as np
import pickle
from PIL import Image
import time
from f_predict import predict
from fliePredictor import predict_image

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
    
    def img_box_lk(self,img2,max_arce,min_arce):
        '''
        通过轮廓画框，加上判断函数
        return  画框图像，轮廓坐标
        '''
        
        kxywh = []
        kxl = []
        #imgs = img2.copy()
        img = self.eroded_and_dilated_img(img2)
        imgss = img.copy()
        imgss = cv2.cvtColor(imgss, cv2.COLOR_GRAY2RGB)
        cv2.imshow("DB",img)
        contours, _ = cv2.findContours(img,cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)  
        for contour in contours:  
            x, y, w, h = cv2.boundingRect(contour) 
            arce = w*h 
            if max_arce> arce >= min_arce:
                
                x, y, w, h = int(x), int(y), int(w), int(h)  
                x_min, y_min = max(0, x), max(0, y)  
                x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h) 
                kxywh.append([x,y,w,h]) 
                cv2.rectangle(imgss, (x_min, y_min), (x_max , y_max), (0, 255, 0), 2)
                cv2.rectangle(img2, (x_min, y_min), (x_max , y_max), (0, 255, 0), 2)
                cv2.imshow("xie",imgss)

        kxl = self.find_and_sort_on_same_line(kxywh)
        return kxl
    
    def img_box_lk_ocr(self,img,max_arce,min_arce,txt):
        '''
        和上面的img_box_lk 作用相同加入排须
        通过轮廓画框，加上判断函数
        return  画框图像，轮廓坐标
        '''
        kxywh = []
        k_ls = []
        k_ls1 = []
        img = self.eroded_and_dilated_img(img)
        contours, _ = cv2.findContours(img,cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)  
        for contour in contours:  
            x, y, w, h = cv2.boundingRect(contour) 
            arce = w*h 
            if max_arce> arce >= min_arce:
                x, y, w, h = int(x), int(y), int(w), int(h)  
                kxywh.append([x,y,w,h])
                x_min, y_min = max(0, x), max(0, y)  
                x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h) 
        kxl = self.find_and_sort_on_same_line(kxywh)
        
        inner_list=[]    
        # print(kxl)
        if len(kxl) > 0: 
            inner_list = kxl[0]   
            for i, sublist in enumerate(inner_list):    
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
        kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # 需要调试
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
        # 腐蚀 ---> 白色前 ，黑色背景
        eroded = cv2.erode(thresh, kernel_1)
        dilated = cv2.dilate(eroded, kernel_1)
        eroded = cv2.erode(dilated, kernel_1)
        dilated = cv2.dilate(eroded, kernel_1)
        
        #dilated = cv2.dilate(dilated, kernel)
        return dilated
    
    def list_clean(self, list):
        processed_list = []
        prev_x = None
        
        for point in list:  
            if prev_x is None or abs(point[0] - prev_x) > 100:  
                processed_list.append(point)  
                prev_x = point[0] 

        return processed_list
    
    def list_all_claen (self,list):
        list_ls = []
        for i in range(len(list)):
            cl = self.list_clean(list[i])
            list_ls.append(cl)

        return list_ls
    
    def boxes_all_claen (self,img,list):
        list_ls = []
        for i in range(len(list)):
            cl = self.boxes_and_img_save(img,list[i])
            list_ls.append(cl)
        return list_ls
    
    def boxes_and_img_save(self,img, list):    
        """    
        boxs : [ (x, y, w, h,label), ...]    
        img : 图像    
        output_dir : 保存图像的目录    
        返回 ：在img 上画框（但不返回），并截图保存到output_dir    
        """  
        ls = []  
        for box in list:    
            x, y, w, h= box    
            x, y, w, h = int(x), int(y), int(w), int(h)  
            x_min, y_min = max(0, x), max(0, y)  
            x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)  
            cropped_img = img[y_min:y_max, x_min:x_max] 
            # 这个是需要修改的
            label = predict_image(cropped_img)
        #     ls.append([x,y,w,h,label])
        # return ls
            
    class svm_predict ():

        def deskew(self,img):
            """去偏斜处理"""
            m = cv2.moments(img)
            if abs(m['mu02']) < 1e-2:
                return img.copy()
            skew = m['mu11'] / m['mu02']
            M = np.float32([[1, skew, -0.5 * img.shape[1] * skew], [0, 1, 0]])
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
            return img


        def preprocess_hog(self,images, max_len=None):
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

            if max_len is not None:
                hog_descriptors = np.array([np.pad(desc, (0, max_len - len(desc)), 'constant') for desc in hog_descriptors],
                                        dtype=np.float32)
            else:
                hog_descriptors = np.array(hog_descriptors, dtype=np.float32)

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


        def predict(self,image_path, model, deskew_func, hog_func, pca, max_len):
            """推理"""
            start_time = time.time()  # 开始计时
            img_array = self.preprocess_image(image_path)
            img = deskew_func(img_array)
            img_hog = hog_func([img], max_len)
            img_hog_pca = pca.transform(img_hog)
            prediction = model.predict(img_hog_pca)
            end_time = time.time()  # 结束计时
            print(f"Processing time: {end_time - start_time} seconds")
            return prediction[0]

                

if __name__ == '__main__':  

    # ------这个程序需要定义的参数-------------------- #
    # min-arce 和max_arce 单个字符最大面积和最小面积
    # 现阶段需要常修改的参数
    # start_angle 起始角度 end_angle
    # ---------------------------------------------- #
    # a  3000   50000
    # b  4000    30000
    #b1 6000    21000
    #c 1000  2000
    #c1 7000   30000
    min_arce = 3000
    max_arce = 30000
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
    dir_origin_path = ".//testimg//testa//b_img//"
    output_dir = ".//zi//"
    image_1 = ".//testimg//testa//b_img//image007.jpg" 

    dect_ocr  = macth_ocr()
    gy = public_img()

    img_names = os.listdir(dir_origin_path)  
    
    image_1 = cv2.imread(image_1)
    img_path = "reslute.jpg"
    st = time.time()
    c_img,rn_img = dect_ocr.out_image(start_angle,end_angle,maxCount,scoreThreshold, iouThreshold, minArea,image,image_1)              

    cv2.imshow("test111",cv2.resize(c_img,(640,320)))
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imshow("result1",rn_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    thr_img_list = gy.img_box_lk(rn_img,max_arce,min_arce)
    clearn_list = gy.list_all_claen(thr_img_list) 
     
    thr_img_box  = gy.list_to_boxes_in_img(rn_img,clearn_list)
    
    all_list = gy.boxes_all_claen(rn_img,clearn_list)
    end = time.time()
    print (end-st)
    #
    
    cv2.imshow("result2",thr_img_box)
    cv2.waitKey(0)



    #单张测试
    #  
    # for img_name in img_names:
    #     image_path = os.path.join(dir_origin_path, img_name)         
    #     tmpe = cv2.imread(image_path)
    #     c_img,r_img = dect_ocr.out_image(start_angle,end_angle,maxCount,scoreThreshold, iouThreshold, minArea,image,tmpe)                 
    #     cv2.imwrite(img_path,r_img)