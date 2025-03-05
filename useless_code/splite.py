import os
import shutil
def get_file_names_without_extension(path):
    file_names = []
    # 遍历路径下的所有项
    for item in os.listdir(path):
        full_item_path = os.path.join(path, item)
        # 判断是否为文件
        if os.path.isfile(full_item_path):
            # 分离文件名与扩展名
            name, _ = os.path.splitext(item)
            file_names.append(name)
    return file_names

def split_list_by_ratio(input_list):
    n = len(input_list)
    # 总比例
    total_ratio = 7 + 2 + 1
    # 计算前三个部分的长度，最后一个部分用剩余的元素
    len1 = int(n * 7 / total_ratio)
    len2 = int(n * 2 / total_ratio)
    len3 = n - len1 - len2 # 确保总数一致

    part1 = input_list[:len1]
    part2 = input_list[len1:len1+len2]
    part3 = input_list[len1+len2:]

    return part1, part2, part3


def move_files_by_name(file_list, source_path, target_path):
    # 如果目标路径不存在，则创建
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # 遍历 source_path 下的所有文件
    for filename in os.listdir(source_path):
        full_source_path = os.path.join(source_path, filename)
        if os.path.isfile(full_source_path):
            # 分离文件名和扩展名
            name, ext = os.path.splitext(filename)
            # 判断文件名是否在 file_list 中
            if name in file_list:
                full_target_path = os.path.join(target_path, filename)

                shutil.move(full_source_path, full_target_path)
                print(f"移动文件: {filename} 到 {target_path}")



path = "LoadData/Kvasir-SEG/images"
result = get_file_names_without_extension(path)
print("文件名列表：", result)
p1, p2, p3 = split_list_by_ratio(result)

print("第一部分：", p1)
print("第二部分：", p2)
print("第三部分：", p3)
print("第一部分：", len(p1))
print("第二部分：", len(p2))
print("第三部分：", len(p3))



move_files_by_name(p2,'LoadData/Kvasir-SEG/images',
                   'LoadData/Kvasir-SEG/val/img')

move_files_by_name(p2,'LoadData/Kvasir-SEG/masks',
                   'LoadData/Kvasir-SEG/val/mask')

move_files_by_name(p3,'LoadData/Kvasir-SEG/images',
                   'LoadData/Kvasir-SEG/test/img')

move_files_by_name(p3,'LoadData/Kvasir-SEG/masks',
                   'LoadData/Kvasir-SEG/test/mask')


