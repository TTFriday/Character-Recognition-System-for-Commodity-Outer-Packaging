import cv2
from f_predict import predict

image_path = 'E://ocrnow//OCR_Detect0611//testimg//yao//data//0//00x1.png'
image = cv2.imread(image_path)
label = predict(image)
print(f"Predicted label: {label}")
