from paddleocr import PaddleOCR, draw_ocr
import glob
import cv2 
# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
# 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
import numpy as np


def getVProjection(img,rect_roi):
    # 这个作用是判断行数
    #  
    # rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotated = img.copy()
    ret, rotated = cv2.threshold(rotated, 125, 255, cv2.THRESH_BINARY)  
    rows,cols=rotated.shape
    ver_list=[0]*cols
    for j in range(cols):
        for i in range(rows):
            if rotated.item(i,j)==0:
                ver_list[j]=ver_list[j]+1
        '''
        对ver_list中的元素进行筛选，可以去除一些噪点
        '''
        ver_arr=np.array(ver_list)
        ver_arr[np.where(ver_arr<1)]=0
        ver_list=ver_arr.tolist()

        #绘制垂直投影
        img_white=np.ones(shape=(rows,cols),dtype=np.uint8)*255
        for j in range(cols):
            pt1=(j,rows-1)
            pt2=(j,rows-1-ver_list[j])
            cv2.line(img_white,pt1,pt2,(0,),1)
    cv2.imshow('chui',img_white)
    cv2.imwrite('chui.jpg',img_white)
    vv_list=get_vvList(ver_list)
    for i in vv_list:
        img_ver=rect_roi[:,i[0]:i[-1]]
        cv2.imshow("dange",img_ver)
        cv2.imwrite('dange.jpg',img_ver)
        cv2.waitKey(0)
        
        # cz(img_ver)
    

def cz(img):

    rotated_img1 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # rotated_img1 = eroded_and_dilated_img_heng(rotated_img1)
    # img_lk=rotated_img1.copy()
    # contours, hierarchy = cv2.findContours(img_lk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    # img_lk = cv2.drawContours(img_lk, contours, -1, (0, 255, 0), 1)  
    # cv2.imshow("1",img_lk)
    
    rows,cols=rotated_img1.shape
    ver_list=[0]*cols
    for j in range(cols):
        for i in range(rows):
            if rotated_img1.item(i,j)==0:
                ver_list[j]=ver_list[j]+1
        '''
        对ver_list中的元素进行筛选，可以去除一些噪点
        '''
        ver_arr=np.array(ver_list)
        ver_arr[np.where(ver_arr<1)]=0
        ver_list=ver_arr.tolist()

        #绘制垂直投影
        img_white=np.ones(shape=(rows,cols),dtype=np.uint8)*255
        for j in range(cols):
            pt1=(j,rows-1)
            pt2=(j,rows-1-ver_list[j])
            cv2.line(img_white,pt1,pt2,(0,),1)
    # cv2.imshow('垂直投影',img_white)
    vv_list=get_vvList(ver_list)
    for i in vv_list:
        img_ver=rotated_img1[:,i[0]:i[-1]]
        adhesive_text_segmentation(img_ver)

        cv2.imshow("dange",img_ver)
        cv2.waitKey(0)
def adhesive_text_segmentation(img):

    '''
    分割图像粘连
    '''
    w,h= img.shape
    print (w,h)



def get_vvList(list_data):
    #取出list中像素存在的区间
    vv_list=list()
    v_list=list()
    for index,i in enumerate(list_data):
        if i>0:
            v_list.append(index)
        else:
            if v_list:
                vv_list.append(v_list)
                #list的clear与[]有区别
                v_list=[]
    return vv_list

def eroded_and_dilated_img_jianhua(img):
    '''
    侵蚀图像
    img ： 图像 输出： 侵蚀后的图像
    判断行
    '''
    # cv2.imshow("test",img)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    # 需要调试
    # a 125 b 125
    _, thresh = cv2.threshold(img, 27, 255, cv2.THRESH_BINARY)  
    # 腐蚀 ---> 白色前 ，黑色背景
    # dilated = cv2.dilate(thresh, kernel_1)
    # eroded = cv2.erode(dilated, kernel)
    
    # cv2.imshow("fu",dilated)
    return thresh

def eroded_and_dilated_img(img):
    '''
    侵蚀图像
    img ： 图像 输出： 侵蚀后的图像
    判断行
    '''
    # cv2.imshow("test",img)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    # 需要调试
    # a 125 b 125
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)  
    # 腐蚀 ---> 白色前 ，黑色背景
    eroded = cv2.erode(thresh, kernel_1)
    dilated = cv2.dilate(eroded, kernel)
    # cv2.imshow("fu",dilated)
    return dilated
    
if __name__ == '__main__':
    ocr = PaddleOCR(use_angle_cls=True, lang="en") # need to run only once to download and load model into memory
    img_path = ".//testimg//testa//model//b.jpg"
    image  = cv2.imread(img_path)

    # img = cv2.imread('.//testimg//testa//model//c.jpg')  
    # cv2.imshow("image",img)
    # GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # en_img = eroded_and_dilated_img(image)
    # cv2.imshow("test",en_img)
    result = ocr.ocr(img_path, cls=True)
    # print(result)
    for line in result:
        for box,text in line:
            # print (box)
            # print (text)
            # print(box[0][0])
            x1,y1 = box[0]
            x2,y2 = box[2]
            text_label= text[0]
            print (text_label)
            # print (line)
            # print('文本:',text,'置信度:',confidence)
            x1 = int(max(0, x1))  
            y1 = int(max(0, y1))  
            x2 = int(min(x2, image.shape[1]))  
            y2 = int(min(y2, image.shape[0]))  
            rect_roi = image[y1:y2, x1:x2]
                # GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            en_img = eroded_and_dilated_img(rect_roi)
            cv2.imshow("test",en_img)
            cv2.imwrite("wei.jpg",en_img)
            getVProjection(en_img,rect_roi)


            
            # 在图像上绘制矩形框，使用蓝色和2像素的线条宽度  
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  
    
    # 显示或保存图像  
    cv2.imshow('Boxes on Image', image)  
    cv2.waitKey(0)  # 等待按键按下  
    cv2.destroyAllWindows() 

