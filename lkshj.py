# # 给定的列表  
# lists = [  
#     [889, 38, 87, 153],    
#     [1027, 33, 103, 158],    
#     [591, 32, 104, 156],    
#     [738, 31, 98, 158],    
#     [156, 30, 105, 157],    
#     [448, 29, 99, 159],    
#     [303, 29, 103, 159],    
#     [13, 28, 102, 159]  
# ]  
  
# # 第一步：找到x+y最小的子列表  
# min_xy_sublist = min(lists, key=lambda x: x[0] + x[1])  
# print("x+y 最小的子列表:", min_xy_sublist)  
  
# # 第二步：获取参考行的y和y+h值  
# reference_y = min_xy_sublist[1]  
# reference_y_plus_h = reference_y + min_xy_sublist[3]  
  
# # 第三步：检查其他子列表是否在同一行  
# same_row_lists = [sublist for sublist in lists if reference_y <= sublist[1] < reference_y_plus_h] 
# sorted_lst = sorted(same_row_lists, key=lambda x: x[0])  
# print("在同一行的子列表:", sorted_lst)  
  
# # 如果只想要判断是否至少有一个子列表在同一行  
# is_at_least_one_same_row = len(same_row_lists) > 1  
# print("是否至少有一个子列表在同一行:", is_at_least_one_same_row)
# # main_list = [  
# #     [1, 2, 3],  
# #     [4, 5, 6],  
# #     [1, 2, 3],  # 这个子列表与第一个相同  
# #     [7, 8, 9],  
# #     [10, 11, 12]  
# # ]  
  
# # list_to_remove = [  
# #     [1, 2, 3],  
# #     [7, 8, 9]  # 我们也想要去掉这个子列表，尽管它不在list_to_remove中直接列出，但我们的目标是去重  
# # ]  
  
# # # 使用列表推导式来创建一个新的列表，其中不包含与list_to_remove中相同的子列表  
# # new_list = [sublist for sublist in main_list if sublist not in list_to_remove]  
  
# # # 但是，这只会去掉list_to_remove中直接列出的子列表  
# # # 为了去重，我们可以使用一个集合来加速查找过程  
# # set_to_remove = set(map(tuple, list_to_remove))  # 将列表转换为不可变类型（如元组）以存储在集合中  
# # new_list = [sublist for sublist in main_list if tuple(sublist) not in set_to_remove]  
  
# # print(new_list)

# def find_and_sort_on_same_line(lst):  
#     ls = []
#     while lst:  
#         # 找到x+y最小的子列表  
        
#         min_xy_sublist = min(lst, key=lambda x: x[0] + x[1])  
          
#         # 获取y和y+h作为参考行  
#         reference_y = min_xy_sublist[1]  
#         reference_y_plus_h = reference_y + min_xy_sublist[3]  
          
#         # 找到在同一行的子列表  
#         same_line_sublists = [sublist for sublist in lst if reference_y-10 <= sublist[1] < reference_y_plus_h]            
#         # 如果找到了在同一行的子列表，则按x排序并添加到list2  
#         if same_line_sublists:  
#             list2 = sorted(same_line_sublists, key=lambda x: x[0])  
#             ls.append(list2)
#             for sublist in same_line_sublists:  
#                 lst.remove(sublist)  
#         else:  
#             ls.append(min_xy_sublist) 
#             lst.remove(min_xy_sublist)  
#     return ls  
  
# # 给定的列表  
# list = [[1688, 618, 143, 173], [1519, 614, 105, 159], [1372, 614, 107, 157], [853, 611, 106, 157], [999, 610, 102, 159], [482, 609, 103, 159], [1153, 607, 163, 184], [337, 607, 105, 161], [191, 607, 106, 161], [48, 607, 105, 160], [627, 601, 189, 191], [1065, 383, 103, 157], [915, 382, 106, 157], [627, 381, 105, 157], [774, 380, 98, 159], [192, 379, 106, 157], [485, 378, 99, 160], [340, 377, 102, 161], [49, 376, 104, 160], [927, 155, 86, 155], [1064, 151, 104, 159], [628, 150, 105, 156], [775, 149, 99, 157], [193, 148, 105, 156], [485, 147, 99, 159], [341, 147, 102, 159], [51, 146, 101, 158]]
  
# # 调用函数处理列表  
# ls = find_and_sort_on_same_line(list)  
# print (ls)
# 注意：由于lst在函数内部被修改，这里的lst已经是空的或者包含剩余未处理的子列表

data = [[[0, 3, 97, 158], [138, 5, 105, 156], [286, 4, 102, 159], [430, 4, 99, 159], [573, 7, 105, 156], [720, 6, 99, 157], [872, 12, 86, 154], [1009, 8, 104, 158]]]  
data_str = "20130102"  
  
# 由于只有一个外层列表，我们先获取内部的列表  
inner_list = data[0]  
  
# 遍历内部列表和data_str的字符  
for i, sublist in enumerate(inner_list):  
    # 在每个子列表中添加对应的字符（假设data_str足够长）  
    sublist.append(data_str[i])  
  
# 打印结果以验证  
print(data)