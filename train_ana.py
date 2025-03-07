import pandas as pd


def process_csv(input_file, output_file):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 根据 A 列分组，计算 C 列的平均值
    result = df.groupby('epoch', as_index=False)['train_loss'].mean()

    # 将结果保存为新的 CSV 文件
    result.to_csv(output_file, index=False)
    print(f"结果已保存到 {output_file}")


if __name__ == '__main__':
    # 设置输入和输出文件路径
    input_file = 'ans/train_loss_log_bnet_isic2018.csv'
    output_file = 'ans/bnet_isic2018.csv'

    process_csv(input_file, output_file)