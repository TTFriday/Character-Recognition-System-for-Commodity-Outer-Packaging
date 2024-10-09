import cv2
import os
import random
def eroded_and_dilated_img(img):
        '''
        侵蚀图像
        img ： 图像 输出： 侵蚀后的图像
        '''
        # cv2.imshow("test",img)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 13))
        # 这个也要改
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)  
        # 腐蚀 ---> 白色前 ，黑色背景
        # eroded = cv2.erode(thresh, kernel_1)
        # dilated = cv2.dilate(eroded, kernel)
        
        dilated = cv2.dilate(thresh, kernel)
        eroded = cv2.erode(dilated, kernel_1)
        
        return eroded

def img_box_lk(img,max_arce,min_arce):
        '''
        通过轮廓画框，加上判断函数
        return  画框图像，轮廓坐标
        '''
        kxywh = []
        # img = self.eroded_and_dilated_img(img)
        contours, _ = cv2.findContours(img,cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)  
        for contour in contours:  
            x, y, w, h = cv2.boundingRect(contour) 
            arce = w*h 
            if max_arce> arce >= min_arce:                
                kxywh.append([x,y,w,h])
                x, y, w, h = int(x), int(y), int(w), int(h)  
                x_min, y_min = max(0, x), max(0, y)  
                x_max, y_max = min(img.shape[1], x + w), min(img.shape[0], y + h)  
                cv2.rectangle(img, (x_min, y_min), (x_max , y_max), (0, 255, 255), 3)
        return img,kxywh

def find_and_sort_on_same_line(lst):  
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


image_path = './/testimg//testa//model/c1.jpg'
img = cv2.imread(image_path)
img_en = eroded_and_dilated_img(img)
image_path = 'relsimg.jpg'
cv2.imwrite(image_path,img_en)

img_e1,kxywh = img_box_lk(img_en,40000,5000)
# print (kxywh)
ls = find_and_sort_on_same_line(kxywh)
str_txt1 = 'K20501'
str_txt2 = 'HB21042023'
inner_list = ls[0]
ls_list  = ls[1]
print (ls_list)
processed_list = [] 
pr_list  =[] 
prev_x = None  
prev_y = None
for point in inner_list:  
    if prev_x is None or abs(point[0] - prev_x) > 100:  
        # 保留点  
        processed_list.append(point)  
        prev_x = point[0]  
for points in ls_list:  
    if prev_x is None or abs(point[0] - prev_x) > 100:  
        # 保留点  
        pr_list.append(points)  
        print (pr_list)
        prev_y = point[0] 
# print (len(ls[0]))
# inner_list_2 = ls[1]
# # print (str_txt2[0])    
#             # 遍历内部列表和data_str的字符
# sublist_a =[]
# sublist1_a =[]  
for i, sublist in enumerate(processed_list):  
        sublist.append(str_txt1[i])
        # sublist_a.append(sublist)
        

for i, sublist1 in enumerate(pr_list):  
# 在每个子列表中添加对应的字符（假设data_str足够长）  
        sublist1.append(str_txt2[i]) 


print (processed_list)
print(pr_list)
cv2.imshow("text",img_e1)
cv2.waitKey(0)

