import os

def validate_file_names(path1, path2):
    # 获取path1中所有文件的文件名（不包含扩展名）
    files1 = {os.path.splitext(f)[0] for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))}
    # 获取path2中所有文件的文件名（不包含扩展名）
    files2 = {os.path.splitext(f)[0] for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))}

    if files1 != files2:
        raise ValueError(f"两个路径中的文件名不完全相等。\n路径1中的文件名：{files1}\n路径2中的文件名：{files2}")
    else:
        print("验证通过，两个路径中的文件名完全相等。")

if __name__ == "__main__":
    path1 = 'LoadData/Kvasir-SEG/train/img'
    path2 = 'LoadData/Kvasir-SEG/train/mask'

    try:
        validate_file_names(path1, path2)
    except Exception as e:
        print("验证失败：", e)