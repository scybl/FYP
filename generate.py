import random


def random_numbers(x, y, z):
    # 检查 z 是否不大于范围内数字的数量
    total_numbers = y - x + 1
    if z > total_numbers:
        raise ValueError("z 不能大于范围内数字的数量")

    # 从范围 [x, y] 中随机选取 z 个不重复的数字
    numbers = random.sample(range(x, y + 1), z)
    return numbers


if __name__ == '__main__':
    # 示例输入
    x = 1
    y = 612
    z = 122

    result = random_numbers(x, y, z)
    print(result)