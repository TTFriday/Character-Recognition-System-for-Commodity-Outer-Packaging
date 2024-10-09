import cv2
import numpy as np
import pickle
from PIL import Image
import time

def deskew(img):
    """去偏斜"""
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * img.shape[1] * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def preprocess_hog(images, max_len=None):
    """计算HOG"""
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

def preprocess_image(image_array, size=(28, 28)):
    """规范化 固定尺寸的二值图像"""
    img_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    mean_pixel = np.mean(img_array)
    # 判断背景颜色
    if mean_pixel < 128:
        binary_array = np.where(img_array > 128, 0, 255).astype(np.uint8)
    else:
        binary_array = np.where(img_array < 128, 0, 255).astype(np.uint8)
    coords = np.column_stack(np.where(binary_array < 255))
    if coords.size == 0:
        raise ValueError("Image is entirely white.")
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped_img = Image.fromarray(img_array).crop((x_min, y_min, x_max + 1, y_max + 1))
    cropped_img.thumbnail(size, Image.LANCZOS)
    background = Image.new('L', size, 255)
    paste_position = ((size[0] - cropped_img.size[0]) // 2, (size[1] - cropped_img.size[1]) // 2)
    background.paste(cropped_img, paste_position)

    img_array = np.array(background)
    mean_pixel = np.mean(img_array)
    img_array = np.where(img_array < mean_pixel, 255, 0).astype(np.uint8)
    return img_array

def load_model(model_path):
    """加载模型"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def predict_image(image_array, model, deskew_func, hog_func, pca, max_len):
    """推理"""
    # start_time = time.time()  # 开始计时
    img_array = preprocess_image(image_array)
    img = deskew_func(img_array)
    img_hog = hog_func([img], max_len)
    img_hog_pca = pca.transform(img_hog)
    prediction = model.predict(img_hog_pca)
    # end_time = time.time()  # 结束计时
    # print(f"Processing time: {end_time - start_time} seconds")
    return prediction[0]

def predict(image_array):
    """主推理函数"""
    # 模型路径 记得修改
    letters_digits_model_path = "./svm_letters_digits.dat"
    # 加载模型和最大HOG描述符长度以及PCA模型
    letters_digits_model, max_len, pca = load_model(letters_digits_model_path)
    # 预测
    result_letters_digits = predict_image(image_array, letters_digits_model, deskew, preprocess_hog, pca, max_len)
    return result_letters_digits
