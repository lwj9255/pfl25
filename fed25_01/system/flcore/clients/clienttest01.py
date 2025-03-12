import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data

import numpy as np
import time
import torch
import torch.nn as nn
import copy

import math


class clientTest01(object):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)  # 设置随机种子为 0，确保实验可复现

        # 深拷贝传入的全局模型，避免修改原始模型
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm  # 获取使用的算法
        self.dataset = args.dataset  # 获取数据集名称
        self.device = args.device  # 获取设备信息（CPU 或 GPU）
        self.id = id  # 客户端的唯一标识符
        self.save_folder_name = args.save_folder_name  # 保存文件的文件夹名称

        self.num_classes = args.num_classes  # 类别数量
        self.train_samples = train_samples  # 训练样本数量
        self.test_samples = test_samples  # 测试样本数量
        self.batch_size = args.batch_size  # 批量大小
        self.learning_rate = args.local_learning_rate  # 学习率
        self.local_epochs = args.local_epochs  # 本地训练的轮数

        # 检查模型中是否有 BatchNorm 层
        self.has_BatchNorm = False  # 默认没有 BatchNorm 层
        for layer in self.model.children():  # 遍历模型的所有层
            if isinstance(layer, nn.BatchNorm2d):  # 如果某层是 BatchNorm2d
                self.has_BatchNorm = True  # 标记为有 BatchNorm
                break  # 找到后跳出循环

        # 去除与慢训练和慢发送相关的代码，因为它们不再需要
        self.train_slow = None  # 已移除不需要的慢训练参数
        self.send_slow = None  # 已移除不需要的慢发送参数
        self.train_time_cost = {'训练的轮次数': 0, '累计训练所花费的总时间': 0.0}  # 保留训练时间统计
        self.send_time_cost = {'发送的轮次数': 0, '累计发送所花费的总时间': 0.0}  # 保留发送时间统计

        self.loss = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)  # 使用 SGD 优化器
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(  # 学习率调度器，通过指数衰减调整学习率
            optimizer=self.optimizer,  # 前面定义的优化器
            gamma=args.learning_rate_decay_gamma  # 学习率衰减比例，默认是0.99
        )
        self.learning_rate_decay = args.learning_rate_decay  # 学习率是否启用衰减，默认是false

        self.args = args
        self.customized_model = copy.deepcopy(self.model)  # 定制的全局模型初始时等于全局模型
        self.global_mask, self.local_mask = None, None  # 初始化全局掩码和本地掩码

        # 初始化分组参数（用于后续掩码和参数更新）
        self.alpha_params = {
            'feature': {
                'min': args.alpha_min_feat,
                'max': args.alpha_max_feat,
                'k_rate': args.k_rate_feat
            },
            'classifier': {
                'min': args.alpha_min_cls,
                'max': args.alpha_max_cls,
                'k_rate': args.k_rate_cls
            }
        }

        # 参数分组配置（根据实际模型结构调整）
        self.layer_groups = self._init_layer_groups()
        self.previous_params = None  # 初始化前一轮的参数
        self.current_round = 0  # 当前轮次
        self.total_rounds = args.global_rounds  # 总轮次

    def _init_layer_groups(self):
        """根据层类型自动分组"""
        groups = {'feature': [], 'classifier': []}  # 仅保留两个组

        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):  # BN 归为 feature
                groups['feature'].append(name)
            elif isinstance(layer, nn.Linear):
                groups['classifier'].append(name)
        # if(self.id == 1):
        #     print("所有层的分组情况：", groups)  # 输出所有分组的最终结果

        return groups

    def _get_layer_group(self, name):
        """获取参数所属的层分组"""
        module_name = ".".join(name.split(".")[:-1])  # 去掉最后的 .weight 或 .bias，保留层名
        # print(f"检查参数 {name}，解析后模块名：{module_name}")  # 调试信息

        for group, layers in self.layer_groups.items():
            if module_name in layers:
                # if(name == "conv1.weight" and self.id == 1):
                #     print(f"参数 {name} 属于 {group} 组")  # 输出该参数所属的组
                return group

        print(f"警告：未找到参数 {name} 的所属分组，默认返回 'feature'")
        return 'feature'

    def load_train_data(self, batch_size=None):
        """
        加载训练数据
        """
        if batch_size == None:  # 如果没有传入批量大小，使用默认的批量大小
            batch_size = self.batch_size
        # 调用外部函数加载客户端训练数据
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)  # 使用 DataLoader 封装训练数据
        # shuffle: 如果为 True，则在每个 epoch 开始时会将数据集打乱。

    def load_test_data(self, batch_size=None):
        """
        加载测试数据
        """
        if batch_size == None:  # 如果没有传入批量大小，使用默认的批量大小
            batch_size = self.batch_size
        # 调用外部函数加载客户端测试数据
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size=batch_size, shuffle=False)  # 使用 DataLoader 封装测试数据

    def set_parameters(self, global_model, personal_model):
        """
        这里的模型设置方法已根据第一轮和后续轮次区分。
        如果是第一轮，客户端只接收全局模型；
        第二轮及之后，客户端接收全局模型和个性化模型，并根据掩码调整本地模型。
        """
        if self.train_time_cost['训练的轮次数'] == 0:
            # 第一轮：只接收全局模型
            for new_param, old_param in zip(global_model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()  # 全局模型的参数覆盖本地模型
        else:
            # 第二轮及之后：接收全局模型和个性化模型
            if self.local_mask is None:
                print(f"警告：客户端 {self.id} 的 local_mask 为空")

            # 获取全局模型参数字典（包含所有层，包括缓冲区）
            global_dict = global_model.state_dict()
            # 获取本地模型参数字典（包含所有层，包括缓冲区）
            local_dict = self.model.state_dict()

            # 遍历所有全局模型参数（包括缓冲区）
            for name in global_dict:
                # 跳过不需要混合的参数类型
                if name not in self.local_mask:
                    continue  # 保持本地模型原有参数（包含批归一化统计参数）

                # 处理可训练参数：通过掩码混合
                if name in self.local_mask:
                    # 获取设备一致的参数和掩码
                    mask = self.local_mask[name].bool().to(self.device)
                    global_param = global_dict[name].to(self.device)
                    local_param = local_dict[name].to(self.device)

                    # 执行混合：mask=1保留本地，mask=0使用全局
                    local_dict[name] = torch.where(mask, local_param, global_param)

            # 加载更新后的参数到本地模型
            self.model.load_state_dict(local_dict)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        # 把模型设置为评估模式
        self.model.eval()

        # 累积预测正确的样本数量
        test_acc = 0
        # 累积测试样本的总数
        test_num = 0
        # 用于存储模型对每个样本的预测概率分布
        y_prob = []
        # 用于存储每个样本的真实标签
        y_true = []

        # 评估过程不计算梯度
        with torch.no_grad():
            for x, y in testloaderfull:
                # 检查x是否是一个列表
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                # 统计正确的样本数量
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                # 统计测试集总样本数量
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())

                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    def train(self):
        # 确保模型和数据都在正确的设备上
        self.model.to(self.device)

        if self.previous_params is None:
            self.previous_params = {n: p.detach().clone()
                                    for n, p in self.model.named_parameters()}

        trainloader = self.load_train_data()  # 加载训练数据

        start_time = time.time()  # 记录训练开始的时间

        self.model.train()  # 将模型设置为训练模式，训练模式下开启Dropout 和 BatchNorm 的行为

        max_local_epochs = self.local_epochs  # 设定本地训练轮次

        # 本地训练过程
        for epoch in range(max_local_epochs):  # 遍历每个训练周期
            for i, (x, y) in enumerate(trainloader):  # 遍历训练数据
                # enumerate 用于将一个可迭代对象组合为一个索引序列，同时返回元素的索引和值。
                if type(x) == type([]):  # 如果x是一个列表
                    x[0] = x[0].to(self.device)  # 将数据移动到设备上
                else:
                    x = x.to(self.device)  # 将数据移动到设备上
                y = y.to(self.device)  # 将标签移动到设备上

                output = self.model(x)  # 前向传播
                loss = self.loss(output, y)  # 计算损失
                self.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播计算梯度
                self.optimizer.step()  # 更新模型参数

        # 动态计算alpha值（按分组）
        mask_dict = {}  # 修改为字典
        for name, param in self.model.named_parameters():
            group = self._get_layer_group(name)
            params = self.alpha_params[group]

            # 计算动态alpha
            alpha = params['min'] + (params['max'] - params['min']) * ((self.current_round / self.total_rounds) ** params['k_rate'])

            # 计算掩码
            mask_dict[name] = self._calculate_mask(param, self.previous_params[name], alpha)

        # 保存掩码
        self.local_mask = {k: v.clone() for k, v in mask_dict.items()}
        # 输出 local_mask
        # 输出掩码的形状
        # for idx, (key, mask) in enumerate(self.local_mask.items()):
        #     if (idx < 5 and self.id == 1):  # 只打印前5个键的掩码
        #         print(f"参数名称: {key}, 掩码形状: {mask.shape}, 掩码值: {mask}")

        # 更新训练时间相关的统计信息
        self.train_time_cost['训练的轮次数'] += 1  # 训练回合数加1
        self.train_time_cost['累计训练所花费的总时间'] += time.time() - start_time  # 训练时间累加

        # 更新 previous_params，以便在下轮训练中计算变化量
        self.previous_params = {name: param.detach().clone() for name, param in self.model.named_parameters()}

    def _calculate_mask(self, param, previous_param, alpha):
        """
        计算单个参数的掩码
        :param param: 当前参数
        :param previous_param: 上一轮的参数
        :param alpha: 动态计算的 alpha 值
        :return: 掩码（与参数形状一致）
        """
        # 计算参数变化绝对值并展平
        delta = torch.abs(param.detach() - previous_param)
        delta_flat = delta.flatten()

        # 按变化量降序排序
        sorted_values, sorted_indices = torch.sort(delta_flat, descending=True)

        # 计算累积变化占比
        total_change = torch.sum(sorted_values)
        if total_change == 0:  # 如果没有变化，直接返回全 0 掩码，全部都是全局参数
            return torch.zeros_like(delta_flat).reshape_as(delta)

        cumulative_ratio = torch.cumsum(sorted_values, dim=0) / total_change

        # 找到达到阈值 alpha 的最小 k
        mask = (cumulative_ratio >= alpha).float()  # 将布尔张量转换为浮点张量
        k = torch.argmax(mask).item() + 1  # 正确找到第一个满足条件的索引

        # 生成扁平化掩码后 reshape 回原层形状
        flat_mask = torch.zeros_like(delta_flat)
        flat_mask[sorted_indices[:k]] = 1
        return flat_mask.reshape_as(delta)
