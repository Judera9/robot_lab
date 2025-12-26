import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_tensor_heatmap(tensor_dict, figsize=(12, 8), cmap='viridis', 
                       title='Tensor Heatmap', xlabel='Feature Index', 
                       ylabel='Tensor Name', normalize=True):
    """
    绘制一维tensor字典的热图
    
    Args:
        tensor_dict: 字典，key是名称，value是一维tensor或numpy数组
        figsize: 图形大小
        cmap: 颜色映射
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        normalize: 是否对数据进行归一化（0-1范围）
    """
    # 转换所有tensor为numpy数组
    data_matrix = []
    labels = []
    max_length = 0
    
    for name, tensor in tensor_dict.items():
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        elif not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor)
            
        # 确保是一维的
        if tensor.ndim > 1:
            tensor = tensor.flatten()
        elif tensor.ndim == 0:
            # 处理标量值
            tensor = np.array([tensor])
            
        data_matrix.append(tensor)
        labels.append(name)
        max_length = max(max_length, len(tensor))
    
    # 填充所有数组到相同长度
    padded_data = []
    for arr in data_matrix:
        if len(arr) < max_length:
            # 用NaN填充较短的数据
            padded = np.full(max_length, np.nan)
            padded[:len(arr)] = arr
            padded_data.append(padded)
        else:
            padded_data.append(arr)
    
    # 转换为numpy矩阵
    data_matrix = np.array(padded_data)
    
    # 归一化数据（忽略NaN）
    if normalize:
        for i in range(data_matrix.shape[0]):
            row = data_matrix[i]
            valid_mask = ~np.isnan(row)
            if np.any(valid_mask):
                row_valid = row[valid_mask]
                row_min, row_max = row_valid.min(), row_valid.max()
                if row_max > row_min:
                    row[valid_mask] = (row_valid - row_min) / (row_max - row_min)
                else:
                    row[valid_mask] = 0.5
            data_matrix[i] = row
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热图
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(max_length))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    
    # 设置标签
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value')
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 调整布局
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_multiple_tensor_heatmaps(tensor_dict_dict, figsize=(15, 10), 
                                 cmap='viridis', title='Multiple Tensor Heatmaps'):
    """
    绘制多个tensor字典的热图，每个字典作为一个子图
    
    Args:
        tensor_dict_dict: 字典的字典，外层key是子图标题，内层是tensor字典
        figsize: 图形大小
        cmap: 颜色映射
        title: 总标题
    """
    n_subplots = len(tensor_dict_dict)
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
    
    if n_subplots == 1:
        axes = [axes]
    
    for idx, (subplot_title, tensor_dict) in enumerate(tensor_dict_dict.items()):
        ax = axes[idx]
        
        # 处理数据
        data_matrix = []
        labels = []
        max_length = 0
        
        for name, tensor in tensor_dict.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            elif not isinstance(tensor, np.ndarray):
                tensor = np.array(tensor)
                
            if tensor.ndim > 1:
                tensor = tensor.flatten()
            elif tensor.ndim == 0:
                # 处理标量值
                tensor = np.array([tensor])
                
            data_matrix.append(tensor)
            labels.append(name)
            max_length = max(max_length, len(tensor))
        
        # 填充数据
        padded_data = []
        for arr in data_matrix:
            if len(arr) < max_length:
                padded = np.full(max_length, np.nan)
                padded[:len(arr)] = arr
                padded_data.append(padded)
            else:
                padded_data.append(arr)
        
        data_matrix = np.array(padded_data)
        
        # 归一化
        for i in range(data_matrix.shape[0]):
            row = data_matrix[i]
            valid_mask = ~np.isnan(row)
            if np.any(valid_mask):
                row_valid = row[valid_mask]
                row_min, row_max = row_valid.min(), row_valid.max()
                if row_max > row_min:
                    row[valid_mask] = (row_valid - row_min) / (row_max - row_min)
                else:
                    row[valid_mask] = 0.5
            data_matrix[i] = row
        
        # 绘制热图
        im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
        
        # 设置坐标轴和标签
        ax.set_xticks(np.arange(max_length))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(subplot_title)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Tensor Name')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax)
        
        # 旋转x轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return fig, axes

def extract_gru_parameters(gru_layer, layer_idx=0):
    """
    提取GRU层的参数，按门分割
    
    Args:
        gru_layer: PyTorch GRU层
        layer_idx: 层索引（用于多层GRU）
    
    Returns:
        dict: 包含所有参数的字典
    """
    hidden_size = gru_layer.hidden_size
    
    # 提取权重
    weight_ih = gru_layer.weight_ih_l0 if layer_idx == 0 else getattr(gru_layer, f'weight_ih_l{layer_idx}')
    weight_hh = gru_layer.weight_hh_l0 if layer_idx == 0 else getattr(gru_layer, f'weight_hh_l{layer_idx}')
    
    # 提取偏置
    bias_ih = gru_layer.bias_ih_l0 if layer_idx == 0 else getattr(gru_layer, f'bias_ih_l{layer_idx}')
    bias_hh = gru_layer.bias_hh_l0 if layer_idx == 0 else getattr(gru_layer, f'bias_hh_l{layer_idx}')
    
    # GRU的权重是按 [reset, update, new] 顺序连接的
    # 分割权重
    weight_ih_reset, weight_ih_update, weight_ih_new = torch.split(weight_ih, hidden_size, dim=0)
    weight_hh_reset, weight_hh_update, weight_hh_new = torch.split(weight_hh, hidden_size, dim=0)
    
    # 分割偏置
    bias_ih_reset, bias_ih_update, bias_ih_new = torch.split(bias_ih, hidden_size, dim=0)
    bias_hh_reset, bias_hh_update, bias_hh_new = torch.split(bias_hh, hidden_size, dim=0)
    
    return {
        f'layer_{layer_idx}_weight_ih_reset': weight_ih_reset,
        f'layer_{layer_idx}_weight_ih_update': weight_ih_update,
        f'layer_{layer_idx}_weight_ih_new': weight_ih_new,
        f'layer_{layer_idx}_weight_hh_reset': weight_hh_reset,
        f'layer_{layer_idx}_weight_hh_update': weight_hh_update,
        f'layer_{layer_idx}_weight_hh_new': weight_hh_new,
        f'layer_{layer_idx}_bias_ih_reset': bias_ih_reset,
        f'layer_{layer_idx}_bias_ih_update': bias_ih_update,
        f'layer_{layer_idx}_bias_ih_new': bias_ih_new,
        f'layer_{layer_idx}_bias_hh_reset': bias_hh_reset,
        f'layer_{layer_idx}_bias_hh_update': bias_hh_update,
        f'layer_{layer_idx}_bias_hh_new': bias_hh_new,
    }

def compute_gru_statistics(gru_layer, layer_idx=0):
    """
    计算GRU参数的统计量
    
    Args:
        gru_layer: PyTorch GRU层
        layer_idx: 层索引
    
    Returns:
        dict: 包含统计量的字典
    """
    params = extract_gru_parameters(gru_layer, layer_idx)
    stats = {}
    
    for name, param in params.items():
        if 'weight' in name:
            # 计算权重的L2范数（按输出维度）
            l2_norm = torch.norm(param, dim=1)
            stats[f'{name}_l2_norm'] = l2_norm
            
            # 计算权重的均值和标准差
            stats[f'{name}_mean'] = torch.mean(param)
            stats[f'{name}_std'] = torch.std(param)
    
    return stats

def plot_gru_heatmaps(gru_model, title="GRU Parameters Heatmap"):
    """
    绘制GRU模型参数的热图
    
    Args:
        gru_model: 包含GRU层的模型
        title: 图表标题
    """
    # 找到所有的GRU层
    gru_layers = []
    for name, module in gru_model.named_modules():
        if isinstance(module, torch.nn.GRU):
            gru_layers.append((name, module))
    
    if not gru_layers:
        print("No GRU layers found in the model")
        return
    
    # 组织数据用于可视化
    all_weights = {}
    all_stats = {}
    
    for layer_name, gru_layer in gru_layers:
        layer_idx = int(layer_name.split('_')[-1]) if '_' in layer_name else 0
        
        # 提取参数
        params = extract_gru_parameters(gru_layer, layer_idx)
        
        # 只使用权重矩阵进行可视化（2D热图）
        for name, param in params.items():
            if 'weight' in name:
                # 将2D权重矩阵展平为1D用于热图
                all_weights[name] = param.flatten()
        
        # 计算统计量
        stats = compute_gru_statistics(gru_layer, layer_idx)
        all_stats.update(stats)
    
    # 绘制权重矩阵的热图
    print("=== GRU Weight Matrices (Flattened) ===")
    plot_tensor_heatmap(all_weights, title=f"{title} - Weight Matrices", normalize=True)
    
    # 绘制L2范数
    l2_stats = {k: v for k, v in all_stats.items() if 'l2_norm' in k}
    if l2_stats:
        print("\n=== GRU Weight L2 Norms ===")
        plot_tensor_heatmap(l2_stats, title=f"{title} - L2 Norms", normalize=True)
    
    # 绘制均值和标准差
    mean_stats = {k: v for k, v in all_stats.items() if '_mean' in k}
    std_stats = {k: v for k, v in all_stats.items() if '_std' in k}
    
    if mean_stats:
        print("\n=== GRU Weight Means ===")
        plot_tensor_heatmap(mean_stats, title=f"{title} - Weight Means", normalize=False)
    
    if std_stats:
        print("\n=== GRU Weight Standard Deviations ===")
        plot_tensor_heatmap(std_stats, title=f"{title} - Weight Std", normalize=False)

def plot_gru_gate_comparison(gru_model, title="GRU Gate Comparison"):
    """
    比较GRU不同门的权重大小
    
    Args:
        gru_model: 包含GRU层的模型
        title: 图表标题
    """
    # 找到第一个GRU层
    gru_layer = None
    for module in gru_model.modules():
        if isinstance(module, torch.nn.GRU):
            gru_layer = module
            break
    
    if gru_layer is None:
        print("No GRU layer found")
        return
    
    # 提取参数
    params = extract_gru_parameters(gru_layer, 0)
    
    # 计算每个门的权重范数
    gate_norms = {}
    for gate in ['reset', 'update', 'new']:
        ih_norm = torch.norm(params[f'layer_0_weight_ih_{gate}'])
        hh_norm = torch.norm(params[f'layer_0_weight_hh_{gate}'])
        gate_norms[f'{gate}_ih'] = torch.tensor([ih_norm])
        gate_norms[f'{gate}_hh'] = torch.tensor([hh_norm])
    
    # 绘制比较图
    plot_tensor_heatmap(gate_norms, title=title, normalize=False, figsize=(10, 4))

# 示例使用
if __name__ == "__main__":
    # 创建示例数据
    tensor_dict = {
        "State": torch.randn(50),
        "Action": torch.randn(30),
        "Reward": torch.randn(20),
        "Next_State": torch.randn(50)
    }
    
    # 绘制单个热图
    plot_tensor_heatmap(tensor_dict, title="Robot Learning Data Heatmap")
    
    # 创建多个字典的示例
    dict_dict = {
        "Training Data": {
            "States": torch.randn(100),
            "Actions": torch.randn(50)
        },
        "Validation Data": {
            "States": torch.randn(80),
            "Actions": torch.randn(40)
        }
    }
    
    # 绘制多个热图
    plot_multiple_tensor_heatmaps(dict_dict, title="Training vs Validation Data")
    
    # GRU分析示例
    import torch.nn as nn
    
    # 创建一个示例GRU模型
    class SampleGRUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1])
    
    # 创建模型并分析
    model = SampleGRUModel()
    
    # 绘制GRU参数热图
    plot_gru_heatmaps(model, title="Sample GRU Model")
    
    # 比较不同门的权重
    plot_gru_gate_comparison(model, title="Gate Weight Comparison")