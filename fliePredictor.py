import cffi
import numpy as np
import cv2

# 定义 CFFI
ffi = cffi.FFI()
# dll 的接口
ffi.cdef("""
    void initialize(const char* modelPath);
    void predictImage(unsigned char* imageData, int width, int height, int mode, int* finalLabel);
    void resetVotes();
""")

# 加载 DLL
dll = ffi.dlopen("ImagePredictor.dll")

# 初始化模型
dll.initialize(b"svm_letters_digits_model.xml")

def predict_image(img, mode=0):
    
    if img is None:
        print("Error: Image not found or unable to load.")
        return
    
    img_data = img.flatten().astype(np.uint8)
    img_ptr = ffi.cast("unsigned char*", img_data.ctypes.data)

    final_label = ffi.new("int*")
    dll.predictImage(img_ptr, img.shape[1], img.shape[0], mode, final_label)

    if final_label[0] != -1:
        print(f"Final Label: {final_label[0]}")
    else:
        print("Prediction did not complete successfully.")

if __name__ == "__main__":
    image_path = r'D:\yu\OCR_Detect0611\zi\7\image_1978918_7.jpg'
    '''mode = 0 禁用投票, mode = 1 启用投票'''
    print (0)
    img = cv2.imread(image_path)
    predict_image(img, mode=1)
    print (1)

