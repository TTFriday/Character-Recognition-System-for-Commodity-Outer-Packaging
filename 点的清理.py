# 原始列表，包含点（假设前两个元素是x和y坐标）  
points_list = [[20, 22, 167, 146], [195, 25, 163, 148], [368, 28, 164, 146], [400, 56, 97, 90], [542, 31, 165, 146], [715, 35, 164, 146], [747, 63, 96, 90], [944, 43, 59, 144]]
  
# 处理列表，移除不满足条件的点  
processed_list = []  
prev_x = None  
for point in points_list:  
    if prev_x is None or abs(point[0] - prev_x) > 100:  
        # 保留点  
        processed_list.append(point)  
        prev_x = point[0]  # 更新prev_x为当前点的x坐标  
print (len(points_list))  
# 打印处理后的列表  
for point in processed_list:  
    print(point)