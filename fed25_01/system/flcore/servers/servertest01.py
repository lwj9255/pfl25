import os
import h5py
import time
import random

import torch
import copy
import numpy as np
from collections import defaultdict

from flcore.clients.clienttest01 import clientTest01
from utils.data_utils import read_client_data


class FedTest01(object):
    def __init__(self, args, times):
        # 存储实验超参数
        self.args = args
        self.device = args.device  # 计算设备（CPU/GPU）
        self.dataset = args.dataset  # 数据集名称
        self.num_classes = args.num_classes  # 分类任务的类别数量
        self.global_rounds = args.global_rounds  # 全局训练轮数
        self.local_epochs = args.local_epochs  # 本地训练轮数
        self.batch_size = args.batch_size  # 本地训练的批次大小
        self.learning_rate = args.local_learning_rate  # 本地学习率

        # 客户端相关参数
        self.num_clients = args.num_clients  # 总客户端数
        self.join_ratio = args.join_ratio  # 参与训练的客户端比例
        self.num_join_clients = int(self.num_clients * self.join_ratio)  # 计算实际参与的客户端数
        self.current_num_join_clients = self.num_join_clients  # 记录当前参与的客户端数

        # 服务器端训练配置
        self.algorithm = args.algorithm  # 选择的联邦学习算法
        self.time_select = args.time_select  # 是否根据时间选择客户端
        self.goal = args.goal  # 训练目标
        self.time_threthold = args.time_threthold  # 时间阈值
        self.save_folder_name = args.save_folder_name  # 结果存储文件夹
        self.auto_break = args.auto_break  # 是否启用自动停止（默认 False）

        # 记录客户端信息
        self.clients = []  # 存储所有客户端实例
        self.selected_clients = []  # 记录本轮选中的客户端

        # 结果记录
        self.rs_test_acc = []  # 记录测试准确率
        self.rs_test_auc = []  # 记录 AUC 指标
        self.rs_train_loss = []  # 记录训练损失
        self.Budget = []  # 记录每一轮训练所花费的时间的

        # 运行时间相关参数
        self.times = times
        self.eval_gap = args.eval_gap  # 评估间隔
        self.client_drop_rate = args.client_drop_rate  # 客户端丢失率

        # 新客户端的相关参数
        self.num_new_clients = args.num_new_clients  # 新加入的客户端数
        self.new_clients = []  # 存储新加入的客户端实例
        self.eval_new_clients = False  # 是否评估新客户端
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new  # 新客户端微调的轮数

        # 动态 α 设定（控制个性化比例）

        self.global_model = copy.deepcopy(args.model)  # 全局模型
        self.personal_model = copy.deepcopy(args.model)  # 个性化模型（每个客户端可能不同）

        self.uploaded_ids = []  # 上传的客户端ID
        self.uploaded_weights = []  # 上传的权重（根据训练样本数）
        self.uploaded_models = []  # 上传的本地模型列表
        self.uploaded_masks = []  # 上传的本地掩码列表

        # **初始化客户端**
        self.set_clients(clientTest01)  # 绑定客户端类，并创建客户端实例

        print(f"\n参与率 / 总客户端数: {self.join_ratio} / {self.num_clients}")
        print("完成创建服务端和客户端")

        # **初始化 epoch 计数器**
        self.epoch = -1  # 记录当前训练的全局轮次

    def set_clients(self, clientObj):
        # 遍历所有客户端并为每个客户端创建一个实例
        for i in range(self.num_clients):  # 遍历客户端数量（从0到num_clients-1）
            # 读取每个客户端的训练数据（is_train=True表示读取训练数据）
            train_data = read_client_data(self.dataset, i, is_train=True)
            # 读取每个客户端的测试数据（is_train=False表示读取测试数据）
            test_data = read_client_data(self.dataset, i, is_train=False)

            # 使用读取到的训练和测试数据长度来初始化每个客户端的实例
            # clientObj 需要传入的参数：args、客户端ID、训练样本数量、测试样本数量
            client = clientObj(self.args,
                               id=i,  # 客户端编号
                               train_samples=len(train_data),  # 训练数据样本数量
                               test_samples=len(test_data))  # 测试数据样本数量

            # 将每个客户端对象添加到客户端列表中
            self.clients.append(client)

            # 打印当前客户端的基本信息（可选）
            # print(f"客户端 {i} 已加入：训练样本数 {len(train_data)}，测试样本数 {len(test_data)}")

    def select_clients(self):
        # 直接使用预定的参与客户端数量，不再支持随机选择
        # current_num_join_clients 已经在初始化的时候预设好了

        # 从客户端列表中选择指定数量的客户端参与
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        # assert语句检查条件是否为真，如果为假，则抛出一个异常终止执行
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            # 判断是否是第一轮
            if client.send_time_cost['发送的轮次数'] == 0:
                # 如果是第一轮，发送初始模型
                client.set_parameters(self.global_model, None)
            else:
                # 如果不是第一轮，发送全局模型和个性化模型
                client.set_parameters(self.global_model, self.personal_model)

            # 记录发送时间
            client.send_time_cost['发送的轮次数'] += 1  # 增加发送的轮次数
            client.send_time_cost['累计发送所花费的总时间'] += 2 * (time.time() - start_time)

    def evaluate(self, acc=None, loss=None):
        # 获取测试和训练集的评估指标
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        # 计算测试集准确率、AUC和训练集平均损失
        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        # 计算每个客户端的准确率和AUC
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        # 如果没有传入acc和loss，直接保存
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        # 打印评估结果
        print("训练集平均损失: {:.4f}".format(train_loss))
        print("测试集平均准确率: {:.4f}".format(test_acc))
        print("测试集平均AUC: {:.4f}".format(test_auc))
        print("每个客户端测试集准确率的标准差: {:.4f}".format(np.std(accs)))
        print("每个客户端测试集AUC的标准差: {:.4f}".format(np.std(aucs)))

    def test_metrics(self):
        # 每个客户端的样本数量
        num_samples = []
        # 每个客户端预测正确的样本数量
        tot_correct = []
        # 每个客户端的AUC值
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        # 不需要处理新客户端，直接获取训练集指标
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def receive_models(self):
        assert (len(self.selected_clients) > 0)  # 选中的客户端数量大于0才会执行

        active_clients = random.sample(self.selected_clients,
                                       int((1 - self.client_drop_rate) * self.current_num_join_clients))

        # 清空已上传的客户端ID、模型、权重等信息
        self.uploaded_ids = []  # 上传的客户端ID
        self.uploaded_weights = []  # 聚合时的权重
        self.uploaded_models = []  # 上传的模型列表
        self.uploaded_masks = []  # 上传的掩码列表

        tot_samples = 0  # 总样本数量

        for client in active_clients:

            try:  # 计算客户端平均每轮花费时间：训练+发送
                client_time_cost = client.train_time_cost['累计训练所花费的总时间'] / client.train_time_cost[
                    '训练的轮次数'] + \
                                   client.send_time_cost['累计发送所花费的总时间'] / client.send_time_cost[
                                       '发送的轮次数']
            except ZeroDivisionError:  # 捕获 ZeroDivisionError，即如果训练轮数或者发送轮数为 0，则将 client_time_cost 设置为 0，避免除以零的错误
                client_time_cost = 0

            if client_time_cost <= self.time_threthold:  # time_threthold：慢客户端的时间阈值，当客户端训练时间超过该值时，会被认为是慢客户端并可能被丢弃，默认为10000

                tot_samples += client.train_samples  # 累加客户端的训练样本数量

                # 上传客户端信息
                self.uploaded_ids.append(client.id)  # 上传客户端ID
                self.uploaded_weights.append(client.train_samples)  # 根据样本数量计算权重
                self.uploaded_models.append(client.model)  # 上传本轮训练后的本地模型

                # 上传本地掩码（即客户端训练过程中计算的掩码）
                self.uploaded_masks.append(client.local_mask)  # 上传客户端的本地掩码

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples  # 计算出了模型聚合时的权重占比

    import torch
    from collections import defaultdict

    def aggregate_dual_models(self):
        """双模型聚合核心逻辑（基于参数级别的聚合）"""
        # print("当前选中的客户端：", [client.id for client in self.selected_clients])

        # 初始化全局和个性化参数的存储容器
        global_params = defaultdict(lambda: torch.zeros_like(param))  # 存储全局参数
        personal_params = defaultdict(lambda: torch.zeros_like(param))  # 存储个性化参数
        global_weights = defaultdict(float)  # 存储全局参数的总权重
        personal_weights = defaultdict(float)  # 存储个性化参数的总权重

        # 第一阶段：参数分类（按参数级别进行分类）
        for client in self.selected_clients:
            mask = client.local_mask  # 客户端上传的掩码
            model_params = client.model.state_dict()  # 获取客户端的模型参数
            client_samples = client.train_samples  # 客户端样本数

            for key, param in model_params.items():
                if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                    continue  # 跳过批归一化层的统计参数

                param_mask = mask[key].bool()  # 将掩码转换为布尔类型
                param = param.to(self.device)  # 将参数转移到正确设备
                param_mask = param_mask.to(self.device)  # 将掩码转移到正确设备

                # 使用布尔掩码直接分类参数
                global_mask = ~param_mask  # 全局参数掩码（取反）
                personal_mask = param_mask  # 个性化参数掩码

                # 按掩码加权累加参数
                global_params[key] += param * global_mask * client_samples
                personal_params[key] += param * personal_mask * client_samples

                # 累加权重
                global_weights[key] += global_mask.sum().item() * client_samples
                personal_weights[key] += personal_mask.sum().item() * client_samples

            # print(f'客户端 {client.id} 参数分类完毕')

        # 第二阶段：加权聚合全局和个性化参数
        for key in global_params:
            if global_weights[key] > 0:
                global_params[key] /= global_weights[key]  # 对全局参数进行加权平均
            if personal_weights[key] > 0:
                personal_params[key] /= personal_weights[key]  # 对个性化参数进行加权平均

        # 将聚合后的参数加载到全局模型和个性化模型
        global_dict = self.global_model.state_dict()  # 获取全局模型的参数
        personal_dict = self.personal_model.state_dict()  # 获取个性化模型的参数

        for key in global_dict:
            if key in global_params:
                global_dict[key] = global_params[key]  # 更新全局模型的参数
        for key in personal_dict:
            if key in personal_params:
                personal_dict[key] = personal_params[key]  # 更新个性化模型的参数

        self.global_model.load_state_dict(global_dict)  # 更新全局模型
        self.personal_model.load_state_dict(personal_dict)  # 更新个性化模型

    def train(self):
        for i in range(self.global_rounds + 1):
            self.epoch = i  # 更新当前的 epoch 值
            s_t = time.time()  # 记录当前时间，计算训练时间
            self.selected_clients = self.select_clients()  # 这里随机参与率默认为false，每轮参与比例为1，因此这里就是所有客户端的序列

            self.send_models()  # 第一轮时直接发送初始模型，后面每一轮都发送全局模型+定制模型

            if i % self.eval_gap == 0:  # eval_gap：评估的轮次间隔，默认为1，也就是每轮都评估一次
                print(f"\n-------------轮次: {i}-------------")
                print("\n评估个性化模型")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_dual_models()  # 修改为双模型聚合方法

            self.Budget.append(time.time() - s_t)
            print('-' * 25, '本轮时间花费', '-' * 25, self.Budget[-1])

        print("\n最佳精度.")
        print(max(self.rs_test_acc))

        print("\n每回合的平均时间成本.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))
