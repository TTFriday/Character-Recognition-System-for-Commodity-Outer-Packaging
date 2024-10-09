import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtWidgets import QMainWindow
from Ui_InterFace import Ui_BottleDetect
from pyzbar.pyzbar import decode
import numpy as np
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image
import time
# from img_to_ocr_class import Match
# from py_demo_class import Detec_Vague, Matcher
from class_py_demo import macth_ocr,Matcher
import cv2
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class BottleDetec(Ui_BottleDetect, QMainWindow):
    imgfilepath = ''
    temfilepath = ''
    imgFlag = False
    temFlag = False
    def __init__(self):
        super(BottleDetec, self).__init__()
        self.setupUi(self)
        self.functions_connect()
        self.setUi()
        self.show()
    
    def setUi(self):
        self.setWindowTitle("Text detection system")
        
    def functions_connect(self):
        self.action_openImg.triggered.connect(self.openimg)
        self.action_executive.triggered.connect(self.detectimg)
        self.action_openTem.triggered.connect(self.opentem)

    def openimg(self):
        filename, filetype = QFileDialog.getOpenFileName(self, 'Please choose an image','D:', 'Image Files (*.png *.jpg *.jpeg *.bmp *.gif)')
        self.imgfilepath = filename
        print("filename", filename)
        print("filetype", filetype)
        if not filename is  None:
            self.imgFlag = True
         # 创建场景
        scene = QGraphicsScene(self)

        # 加载图像文件
        pixmap = QPixmap(filename)  #传入图像路径
        if not pixmap.isNull():
            item = QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.gps_img.setScene(scene)
            self.gps_img.fitInView(item, 2)  # 图像适应视图
            
    def opentem(self):
        filename, filetype = QFileDialog.getOpenFileName(self, 'Please choose an image','D:', 'Image Files (*.png *.jpg *.jpeg *.bmp *.gif)')
        self.temfilepath = filename
        print("filename", filename)
        print("filetype", filetype)
        if not filename is None:
            self.temFlag = True
    
    def detectimg(self):
        if self.imgFlag == True and self.temFlag:
            # ---------------- 需要给定的参数---------------------- #
            #  
            # ---------------------------------------------------- #
            maxCount = 189
            scoreThreshold = 0.5
            iouThreshold = 0.4
            
            start_angle = -40
            end_angle = 50
            minArea = 256
            methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                    'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
            template_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','L','A']
            template_folder_path = './/testimg//yao//data//' 
            image = cv2.imread(self.temfilepath, cv2.IMREAD_GRAYSCALE)
            tmp = cv2.imread(self.imgfilepath)
            # tmpe = cv2.imread(tmp)
            dll_path = './templatematching_ctype.dll' # 模板匹配库路径
            # setting_num = 40 # 单张图片的宽 后面改为有效框的平均宽
            # matcher = Matcher(dll_path, maxCount, scoreThreshold, iouThreshold, angle, minArea)
            # st  = time.time()
            detec_vague = macth_ocr()
            cap,r_img = detec_vague.out_image(start_angle,end_angle,maxCount,scoreThreshold, iouThreshold, minArea,image,tmp)
            # ocr 识别指定区域的内容
            outlist = detec_vague.outputlist(r_img)  
            # 写入图像上  
            w_img = detec_vague.write_img(tmp,outlist)
            # match = Match()
            
            # templates = match.get_templates(template_folder_path, template_chars)
            # r_image,label = match.get_ocr_string(templates ,r_img) 
            # # cv2.imshow('Image with Text', r_image)
            # # cv2.waitKey(0)
            # list_clean = match.non_max_suppression(label) 
            # sorted_boxes = match.group_and_sort_boxes(list_clean)  
            # # print (sorted_boxes)
            # Strlist = match.list_vsstr(sorted_boxes,setting_num)
            # last_image = match.img_to_str(tmpe,Strlist)
            # end = time.time()
            text = []
            str = detec_vague.show_code(tmp)
            str_qt = detec_vague.show_Qrcode(tmp)
            text.append(str)
            text.append(str_qt)
            print(text)
            self.led_num.setText(outlist[0]) 
            self.led_ProDay.setText(outlist[1])
            self.led_ValDay.setText(outlist[2])
            self.led_code.setText(text[0])
            self.led_Qcode.setText(text[1])
            height, width, channel = tmp.shape
            bytes_per_line = 3 * width
            pixmap = QPixmap.fromImage(QImage(tmp.data, width, height, bytes_per_line, QImage.Format_RGB888))
            # 创建 QGraphicsScene
            scene = QGraphicsScene()

            # 将图像添加到 QGraphicsScene 中
            item = QGraphicsPixmapItem(pixmap)
            scene.addItem(item)

            # 将 QGraphicsScene 设置为 gps_img 控件的场景
            self.gps_img.setScene(scene)

            # 使图像适应视图
            self.gps_img.fitInView(item, Qt.KeepAspectRatio)
            # 记录检测数据
            self.teb_recode.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            self.teb_recode.append("Product number：" + outlist[0])
            self.teb_recode.append("Date of manufacture：" + outlist[1])
            self.teb_recode.append("Valid until：" + outlist[2])
            self.teb_recode.append("Barcodes：" + text[0])
            self.teb_recode.append("QR code：" + text[1])
            self.teb_recode.append("-"*10)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    bottledetec = BottleDetec()
    sys.exit(app.exec_())