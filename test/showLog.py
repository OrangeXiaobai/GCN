import re
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 日志文件路径
    log_files = {
        'ACT-GCN': '../work_dir/ntu120/actcgn/actcgn_1/log.txt',
        'CTR-GCN': '../work_dir/ntu120/ctrgcn/ctrgcn_3/log.txt',
        '2s-AGCN': '../work_dir/ntu120/agcn/agcn_1/log.txt',
        'base-GCN': '../work_dir/ntu120/basegcn/basegcn_1/log.txt',
        # 'ACT-GCN_2': '../work_dir/ntu120/actcgn/actcgn_2/log.txt',
    }

    # 用于存储每个模型的 Top1 数据
    data = {
        'ACT-GCN': {},
        'CTR-GCN': {},
        '2s-AGCN': {},
        'base-GCN': {},
        # 'ACT-GCN_2': {},
    }

    # 正则表达式提取 Eval epoch 和 Top1
    epoch_pattern = r"Eval epoch: (\d+)"
    top1_pattern = r"Top1: (\d+\.\d+)%"

    # 从日志中提取所有轮次的 Top1 数据
    for model, filepath in log_files.items():
        with open(filepath, 'r') as f:
            log_lines = f.readlines()

            current_epoch = None
            for line in log_lines:
                # 提取轮次
                epoch_match = re.search(epoch_pattern, line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))

                # 提取 Top1
                top1_match = re.search(top1_pattern, line)
                if top1_match and current_epoch is not None:
                    data[model][current_epoch] = float(top1_match.group(1))

    # 检查提取的数据
    for model in data:
        print(f"Model {model}: {data[model]}")

    # 绘制折线图
    plt.figure(figsize=(10, 6))

    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd']
    for i, (model, top1_dict) in enumerate(data.items()):
        # 按轮次排序并提取数据
        epochs = sorted(top1_dict.keys())
        top1_values = [top1_dict[epoch] for epoch in epochs]

        # 绘制折线
        plt.plot(epochs, top1_values, marker='o', color=colors[i], label=f'{model}')

        # 找到最大值及其对应的轮次
        max_top1 = max(top1_values)
        max_epoch = epochs[top1_values.index(max_top1)]

        # 在最大值处添加三角形标记
        plt.scatter(max_epoch, max_top1, marker='^', s=100, color=colors[i], label=f'Max Top1 {max_top1}')

        # 在三角形旁边添加数值标签
        plt.text(max_epoch, max_top1 + 0.5, f'{max_top1:.2f}',
                 ha='center', va='bottom', fontsize=10, color=colors[i])

    # 设置图形属性
    plt.xlabel('Epoch')
    plt.ylabel('Top1 Accuracy (%)')
    plt.title('Top1 Accuracy Across Epochs for Models ACT-GCN, CTR-GCN, 2s-AGCN, base-GCN')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图片
    plt.tight_layout()
    plt.savefig('./top1.png', dpi=800)

    print("saved to 'top1.png'")