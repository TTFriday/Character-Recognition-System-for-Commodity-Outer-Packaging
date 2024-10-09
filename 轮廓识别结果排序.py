# 原始列表  
lists = [  
    [889, 38, 87, 153],    
    [1027, 33, 103, 158],    
    [591, 32, 104, 156],    
    [738, 31, 98, 158],    
    [156, 30, 105, 157],    
    [448, 29, 99, 159],    
    [303, 29, 103, 159],    
    [13, 28, 102, 159]  
]  
  
# 初始化最小和为无穷大，以及对应的子列表  
min_sum = float('inf')  
min_list = None  
  
# 遍历每个子列表  
for lst in lists:  
    # 计算x+y的和  
    x_plus_y = lst[0] + lst[1]  
    # 如果当前和比已知的最小和小，则更新最小和和对应的子列表  
    if x_plus_y < min_sum:  
        min_sum = x_plus_y  
        min_list = lst  
  
# 输出结果  
print(f"具有最小x+y和的子列表是：{min_list}，其中x+y的和为：{min_sum}")