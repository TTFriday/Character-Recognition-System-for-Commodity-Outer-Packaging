# by yu 0611
# 应付多数的存在
 
# 完成 0611
# 针对b_img
import cv2
import os
import random
from class_py_demo import macth_ocr

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
    def img_box_lk(self,img,max_arce,min_arce):
        '''
        通过轮廓画框，加上判断函数
        return  画框图像，轮廓坐标
        '''
        
        kxywh = []
        kxl = []
        img = self.eroded_and_dilated_img(img)
        contours, _ = cv2.findContours(img,cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)  
        for contour in contours:  
            x, y, w, h = cv2.boundingRect(contour) 
            arce = w*h 
            if max_arce> arce >= min_arce:
                
                x, y, w, h = int(x), int(y), int(w), int(h)  
                x_min, y_min = max(0, x), max(0, y)  
                x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h) 
                kxywh.append([x,y,w,h]) 
                # cv2.rectangle(img, (x_min, y_min), (x_max , y_max), (255, 255, 0), 2)

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
        contours, _ = cv2.findContours(img,cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)  
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        # 需要调试
        _, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)  
        # 腐蚀 ---> 白色前 ，黑色背景
        eroded = cv2.erode(thresh, kernel_1)
        dilated = cv2.dilate(eroded, kernel)
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



    
    # def list_clean_3d(self,lst_3d):  
    #     processed_list_3d = []  
        
    #     for sublist in lst_3d:  # 遍历三维列表中的每个二维列表  
    #         processed_sublist = []  
    #         current_group = []  
    #         prev_x = None  
            
    #         for hang in sublist:  # 遍历二维列表中的每一行  
    #             for point in hang:  # 遍历每一行中的点  
    #                 if prev_x is None or abs(point[0] - prev_x) > 100:  
    #                     if current_group:  # 如果当前组不为空，则将其添加到processed_sublist中  
    #                         processed_sublist.append(current_group)  
    #                     current_group = [point]  # 开始新的组  
    #                 else:  
    #                     current_group.append(point)  # 将点添加到当前组  
    #                 prev_x = point[0]  
            
    #         # 不要忘记在最后添加最后一个组（如果有的话）  
    #         if current_group:  
    #             processed_sublist.append(current_group)  
            
    #         processed_list_3d.append(processed_sublist)  # 将处理后的二维列表添加到三维列表中  
        
    #     return processed_list_3d  
                

if __name__ == '__main__':  

    # ------这个程序需要定义的参数-------------------- #
    # min-arce 和max_arce 单个字符最大面积和最小面积
    # 现阶段需要常修改的参数
    # start_angle 起始角度 end_angle
    # ---------------------------------------------- #
    min_arce = 8000
    max_arce = 50000
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
    image = cv2.imread('.//testimg//testa//model//a.jpg', cv2.IMREAD_GRAYSCALE)
    dir_origin_path = ".//testimg//testa//a_img//"
    output_dir = ".//zi//"
    image_1 = ".//testimg//testa//a_img//image008.jpg" 

    dect_ocr  = macth_ocr()
    gy = public_img()
    img_names = os.listdir(dir_origin_path)  
    
    image_1 = cv2.imread(image_1)
    img_path = "reslute.jpg"
    c_img,rn_img = dect_ocr.out_image(start_angle,end_angle,maxCount,scoreThreshold, iouThreshold, minArea,image,image_1)                 
    thr_img_list = gy.img_box_lk(rn_img,max_arce,min_arce)
    clearn_list = gy.list_all_claen(thr_img_list)
    thr_img_box  = gy.list_to_boxes_in_img(rn_img,clearn_list)
    print (len(clearn_list))
    cv2.imshow("test",thr_img_box)
    cv2.waitKey(0)



    #单张测试
    #  
    # for img_name in img_names:
    #     image_path = os.path.join(dir_origin_path, img_name)         
    #     tmpe = cv2.imread(image_path)
    #     c_img,r_img = dect_ocr.out_image(start_angle,end_angle,maxCount,scoreThreshold, iouThreshold, minArea,image,tmpe)                 
    #     cv2.imwrite(img_path,r_img)