from paddleocr import PaddleOCR, draw_ocr
import cv2
import os
import random
from class_py_demo import macth_ocr

'''
ocr 识别然后根据输出字符的数量去平均分配框
使用范围 ： 数字字母和中文字符连续
现在将公用和ocr自己用的分开public_img()和b_ocr()
先切成单行
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
    
    def img_box_lk(self,img,max_arce,min_arce):
        '''
        通过轮廓画框，加上判断函数
        return  画框图像，轮廓坐标
        '''
        kxy = []
        kwh = []
        kxywh = []
        img = self.eroded_and_dilated_img(img)
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        contours, _ = cv2.findContours(img,cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)  
        for contour in contours:  
            x, y, w, h = cv2.boundingRect(contour) 
            # print (x,y,w,h)
            arce = w*h 
            if max_arce> arce >= min_arce:
                kxy.append([x,y])
                kwh.append(Rectangle(w,h))  
                
                x, y, w, h = int(x), int(y), int(w), int(h) 
                kxywh.append([x,y,w,h]) 
                x_min, y_min = max(0, x), max(0, y)  
                x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)  
                cv2.rectangle(img, (x_min, y_min), (x_max , y_max), (255, 255, 0), 2)
        return img ,kxywh

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
        已经分行
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)  
        # 腐蚀 ---> 白色前 ，黑色背景
        eroded = cv2.erode(thresh, kernel_1)
        dilated = cv2.dilate(eroded, kernel)

        return dilated
    
    def ocr_img_clean(self,img_path,img):
        
        '''
        ocr 识别提供类别label  
        有变动 注意
        '''
        ls_k = []
        result = ocr.ocr(img_path, cls=True)
        for idx in result:            
            for line in idx:
                txt = line[1][0]
                boxes_x1y1 = line[0][0]
                boxes_x2y2 = line[0][2]
                x1 , y1  = boxes_x1y1
                x2, y2  = boxes_x2y2
                ls_k.append([txt,x1,y1,x2,y2])  
                # print (txt)        
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
            
        #    print (x1,y1,w1,h1)
            for box in list1:
                x,y,w,h = box
                center_x1 = x+w/2
                center_y1 = y+h/2
                if (center_x1 >= x1 and center_x1 <=x1+w1) and (center_y1 >= y1 and center_y1 <= y1+h1): 
                    list[i].append(box)
            sorted_data = sorted(list[i], key=lambda item: item[0]) 
            if len(sorted_data) <= len(txt):
                for ls_i in range(len(list[i])):
                    # print (ls_i)
                    x,y,w,h = sorted_data[int(ls_i)]
                    label = txt[ls_i]
                    list_2[i].append([label,x,y,w,h]) 
            else:
                for ls_i in range(len(txt)):
                    x,y,w,h = sorted_data[int(ls_i)]
                    label = txt[ls_i]
                    list_2[i].append([label,x,y,w,h]) 



       return list_2



class b_ocr():
    def __init__(self): 
        self.public_img = public_img()

    def split_rectangle_into_n_parts(self,x, y, w, h, n_fen,str_txt):  
        '''
        ocr 识别后均匀分割
        '''

        if n_fen <= 0:  
            raise ValueError("n must be a positive integer")  
        new_w = w / n_fen  
        new_rectangles = []  
        for i in range(n_fen):   
            new_x = x + i * new_w   
            new_y = y  
            new_h = h  
            label = str_txt[i]
            new_rectangles.append([label,int(new_x), int(new_y), int(new_w), int(new_h)])    
        return new_rectangles  


    def box_img_list(self,img_path,img): 
        ls_k = []
        result = ocr.ocr(img_path, cls=True)
        for idx in result:      
            for box in idx:
                box_x1y1 = box[0][0]
                box_x2y2 = box[0][2]
                x1,y1 = box_x1y1
                x2,y2 = box_x2y2
                txt  = box[1][0] 
                # print (box_x2y2)         
                
                cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0, 255, 0), 1)
            ls_k.append([x1,y1, x2,y2])
 
                    # cv2.rectangle(img, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 1)             
        return img ,ls_k





if __name__ == '__main__':  
    # ------这个程序需要定义的参数------ #
    # min-arce 和max_arce
    # -------------------------------- #
    min_arce = 10000
    max_arce = 50000

    maxCount = 1
    scoreThreshold = 0.5
    iouThreshold = 0.4
    start_angle = -30
    end_angle = 60
    minArea = 256
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    template_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','年','月','日','L','A']
    dll_path = './templatematching_ctype.dll'
    # template_folder_path = './/testimg//yao//data//' 
    image = cv2.imread('.//testimg//testa//model//a.jpg', cv2.IMREAD_GRAYSCALE)
    # 纯英文加数字 en
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
    dir_origin_path = ".//testimg//testa//a_img//"
    output_dir = ".//zi//"
    dect_ocr  = macth_ocr()
    gy = public_img()
    b_ocr = b_ocr()
    img_names = os.listdir(dir_origin_path)  
    image_1 = ".//testimg//testa//a_img//image009.jpg" 
    image_1 = cv2.imread(image_1)
    img_path = "reslute.jpg"
    c_img,rn_img = dect_ocr.out_image(start_angle,end_angle,maxCount,scoreThreshold, iouThreshold, minArea,image,image_1)                 
    cv2.imwrite(img_path,rn_img)
    # print (1)
    '''
    #box_img_list  是由ocr 提供的坐标信息，对字符结构长度相同的识别良好
    # box_img_list_best
    ls_one ocr 识别 分割后的小框
    ls_two 轮廓识别
    ls_three ocr识别大框
    '''
    img , ls_one = b_ocr.box_img_list(img_path,rn_img)  
    # cv2.imshow("the model",img)
    # cv2.waitKey(0) 

    img,ls_two = gy.img_box_lk(rn_img,max_arce,min_arce)
    ls_three = gy.ocr_img_clean(img_path,rn_img)
    cl_list = gy.line_yy_and_clean(ls_two,ls_three)
    print (ls_two)
    # print (len(ls_two))
    # rn_img = eroded_and_dilated_img(rn_img)
    rn1_img = gy.list_to_boxes_in_img(rn_img,cl_list)
    rn1_image = gy.listone_to_boxes_in_img(rn_img,ls_two)
    # print (cl_list)
    cv2.imshow("the model",rn1_img)
    cv2.waitKey(0)
    # for img_name in img_names:
    #     image_path = os.path.join(dir_origin_path, img_name)         
    #     tmpe = cv2.imread(image_path)
    #     c_img,r_img = dect_ocr.out_image(start_angle,end_angle,maxCount,scoreThreshold, iouThreshold, minArea,image,tmpe)                 
    #     cv2.imwrite(img_path,r_img)
    #     ls_one = box_img_list(img_path,r_img)
    #     r_img = eroded_and_dilated_img(r_img)
    #     boxes_and_img(ls_one,r_img,output_dir,template_chars)




