import numpy as np
import time
import torch
import torch.nn as nn
import copy
from flcore.clients.clientbase import Client


class Client_fed03(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.critical_parameter = None
        self.args = args
        self.current_rounds = 0
        self.global_mask, self.local_mask = None, None

    def train(self):
        trainloader = self.load_train_data()  # 加载训练数据

        start_time = time.time()  # 记录训练开始的时间

        # 记录模型在本地更新之前的状态，用于选择关键参数
        initial_model = copy.deepcopy(self.model)

        self.model.train()  # 将模型设置为训练模式，训练模式下开启Dropout 和 BatchNorm 的行为

        max_local_epochs = self.local_epochs  # 最大本地训练轮次，这里根据文献设置为5

        if self.train_slow:  # 如果设置了train_slow，默认是没设置的
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)  # 训练轮次随机减半

        # 本地训练过程
        for epoch in range(max_local_epochs):  # 遍历每个训练周期
            for i, (x, y) in enumerate(trainloader):  # 遍历训练数据
                # enumerate 用于将一个可迭代对象组合为一个索引序列，同时返回元素的索引和值。
                if type(x) == type([]):  # 如果x是一个列表
                    x[0] = x[0].to(self.device)  # 将数据移动到设备上
                else:
                    x = x.to(self.device)  # 将数据移动到设备上
                y = y.to(self.device)  # 将标签移动到设备上

                if self.train_slow:  # 如果启用慢训练，默认不启动
                    time.sleep(0.1 * np.abs(np.random.rand()))  # 延迟训练，模拟慢训练

                output = self.model(x)  # 前向传播
                loss = self.loss(output, y)  # 计算损失
                self.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播计算梯度
                self.optimizer.step()  # 更新模型参数

                # 检查并处理参数中的 NaN
                for name, p in self.model.named_parameters():
                    if torch.isnan(p.data).any():
                        print(f"警告：客户端 {self.id} 参数 {name} 在训练后含 NaN，已替换为 0")
                        p.data = torch.nan_to_num(p.data, nan=0.0)

        if self.learning_rate_decay:  # 如果启用了学习率衰减，默认没启动
            self.learning_rate_scheduler.step()  # 更新学习率

        # 选择关键参数
        self.global_mask, self.local_mask = self.compute_critical_masks(prevModel=initial_model, model=self.model)

        self.current_rounds += 1

        self.train_time_cost['num_rounds'] += 1  # 训练回合数加1
        self.train_time_cost['total_cost'] += time.time() - start_time  # 训练时间累加

    # 选择关键参数
    def compute_critical_masks(self, prevModel: nn.Module, model: nn.Module):
        r"""
        计算每层参数的掩码，用于标记个性化参数（掩码为1）和全局参数（掩码为0）。

        算法步骤：
          1. 对于每层参数，计算参数变化量 Δθ = param - prevparam。
          2. 利用 |Δθ * param| 作为敏感度度量，将其展平成一维向量。
          3. 按降序排序后，累加这些敏感度值，直到累计和达到所有敏感度总和的 α_current 比例，
             其中 α_current = α_min + (α_max - α_min) * (epoch / global_rounds)^α_k。
          4. 累加达到阈值的前 k 个参数标记为个性化（1），其余标记为全局（0）。
          5. 随着训练轮次增加，α_current增大，个性化参数比例逐渐提高。
        """
        global_mask = []  # 用于存储各层的全局参数掩码（0 表示全局参数）
        local_mask = []  # 用于存储各层的个性化参数掩码（1 表示个性化参数）

        # 计算当前轮次对应的 α_current 阈值
        # α_current = α_min + (α_max - α_min) * (epoch / global_rounds)^α_k
        alpha_current = self.args.alpha_min + (self.args.alpha_max - self.args.alpha_min) * (
                self.current_rounds / self.args.global_rounds) ** self.args.alpha_k

        alpha_current = torch.clamp(torch.tensor(alpha_current), min=1e-6, max=1.0)  # 防止 alpha 过小或过大


        # 遍历前一轮模型和当前模型中每一层的参数
        for (name1, prevparam), (name2, param) in zip(prevModel.named_parameters(), model.named_parameters()):
            # 计算参数变化量 Δθ = 当前参数 - 上一轮参数
            diff = param.data - prevparam.data
            # 计算敏感度度量：|Δθ * 当前参数|，反映该参数的变化重要性
            sensitivity = torch.abs(diff * param.data)
            sensitivity = torch.nan_to_num(sensitivity, nan=0.0)  # 处理可能的 NaN

            # 将敏感度展平为一维向量
            metric = sensitivity.view(-1)
            num_params = metric.size(0)  # 当前层参数的总个数

            # 计算该层敏感度的总和
            total_sum = torch.sum(metric)

            # 如果该层所有敏感度之和为0，则该层所有参数均视为非关键（全局参数）
            if total_sum == 0:
                mask = torch.zeros_like(metric, dtype=torch.int)
            else:
                # 对敏感度按降序排序，sorted_vals为排序后的值，sorted_indices为对应原始索引
                sorted_vals, sorted_indices = torch.sort(metric, descending=True)
                # 计算排序后敏感度的累计和
                cumulative_sum = torch.cumsum(sorted_vals, dim=0)
                # 找到累计和首次达到或超过 (α_current * total_sum) 的位置
                threshold_index = (cumulative_sum >= alpha_current * total_sum).nonzero()
                k = threshold_index[0][0].item() + 1 if len(threshold_index) > 0 else num_params

                # 创建一个全零的一维掩码向量，初始所有元素均为0（全局参数）
                mask = torch.zeros(num_params, dtype=torch.int)
                # 将排序后前 k 个参数的位置（原始索引）置为1，表示这些位置为个性化参数
                mask[sorted_indices[:k]] = 1

            # 将一维掩码恢复为原始敏感度矩阵的形状
            mask = mask.view(sensitivity.shape)
            # 将当前层的个性化掩码添加到 local_mask 列表中（1表示个性化参数）
            local_mask.append(mask.clone().to('cpu'))
            # 全局掩码为 (1 - mask)，即个性化参数标记为0，全局参数标记为1
            global_mask.append((1 - mask).to('cpu'))

        # 清空当前模型的梯度
        model.zero_grad()

        # 返回各层的全局掩码和个性化（本地）掩码
        return global_mask, local_mask

    # 设置模型参数
    def set_parameters(self, model):
        # 判断本地掩码是否存在，如果存在则进行个性化参数更新
        if self.local_mask is not None:  # 如果本地掩码不为空，说明需要使用掩码对参数进行调整
            index = 0  # 初始化索引，用于依次访问每个参数对应的掩码
            # 使用 zip 遍历客户端当前模型和服务端传入模型的所有参数
            # self.model.named_parameters() 返回客户端模型中所有参数的 (name, parameter) 对
            # model.named_parameters() 返回服务端生成的个性化模型参数的 (name, parameter) 对
            for ((name1, client_param), (name2, server_param)) in zip(
                    self.model.named_parameters(),  # 客户端当前模型的参数
                    model.named_parameters()  # 服务端生成的个性化模型参数
            ):
                # 从本地掩码列表中获取当前参数对应的掩码，并将其移动到设备上（如 GPU），转换为浮点类型（0.0 或 1.0）
                mask = self.local_mask[index].to(self.device).float()
                # 计算新的参数：
                # 如果掩码为 1，则保留客户端原有参数 (mask * client_param.data)
                # 如果掩码为 0，则用服务端生成的个性化参数替换 ((1 - mask) * server_param.data)
                client_param.data = mask * client_param.data + (1 - mask) * server_param.data
                index += 1  # 索引递增，处理下一个参数
        else:
            # 如果本地掩码为空，则直接调用父类中定义的 set_parameters 方法进行参数更新
            super().set_parameters(model)

    #     # 更新完参数后，冻结 BN 层，防止其统计量更新导致数值不稳定
    #     self.freeze_bn()
    #
    # def freeze_bn(self):
    #     """
    #     遍历客户端模型的所有模块，如果模块为 BatchNorm2d，
    #     则将其切换到 eval 模式并冻结其参数（即 requires_grad=False），
    #     这样在后续的前向过程中就不会更新其统计量。
    #     """
    #     for m in self.model.modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.eval()  # 切换到 eval 模式
    #             for param in m.parameters():
    #                 param.requires_grad = False






