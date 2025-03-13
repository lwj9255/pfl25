import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from flcore.clients.client_fed25_03_simplified import Client_fed03
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data


class Server_fed03(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # ---- 以下是动态掩码相关超参 ----
        self.uploaded_masks = None
        self.alpha_min = float(args.alpha_min)  # 初始个性化比例下限
        self.alpha_max = float(args.alpha_max)  # 最终个性化比例上限
        self.alpha_k = float(args.alpha_k)  # 个性化比例增长曲线的指数系数

        # 扩散模型训练相关超参数
        self.diff_steps = args.diff_steps  # 前向扩散总步数
        self.noise_schedule = args.noise_schedule  # 噪声调度策略，目前只支持线性
        self.diff_lr = args.diff_lr  # 扩散模型学习率
        self.diff_epochs = args.diff_epochs  # 扩散模型训练轮数
        self.diff_bs = args.diff_bs  # 扩散模型训练批次大小
        self.guidance_weight = args.guidance_weight  # 分类器引导权重（本代码未具体用到）
        self.denoise_steps = args.denoise_steps  # 反向生成时的去噪步数
        self.hidden_dim = args.hidden_dim  # 扩散模型隐藏层维度
        self.latent_dim = args.latent_dim  # 参数潜在编码的压缩维度（本代码中未做压缩）
        self.personalized_models = []  # 存储生成的个性化模型

        # 为扩散模型和潜在编码预留占位符（后续训练中实例化）
        self.diffusion_model = None
        self.latent_encodings = {}  # 存储每个客户端的潜在编码γ

        # 随机选择慢客户端（但现在没开）
        self.set_slow_clients()  # 根据train_slow_rate标记哪些客户端在训练阶段是慢的，根据send_slow_rate标记哪些在发送阶段是慢的

        self.set_clients(Client_fed03)  # 设置客户端

        print(f"\n参与率 / 总客户端数: {self.join_ratio} / {self.num_clients}")
        print("完成创建服务端和客户端")

        self.Budget = []  # 初始化空的 Budget 列表，用于存储时间开销

        # 全局轮次计数器
        self.epoch = -1  # 设置epoch 为 -1，这和self.local_epoch不是同一个

        # 开启 matplotlib 交互模式，使得图表可以动态更新
        plt.ion()

    def train(self):
        for i in range(self.global_rounds + 1):
            self.epoch = i  # 更新当前的 epoch 值
            s_t = time.time()  # 记录当前时间，计算训练时间

            self.selected_clients = self.select_clients()  # 这里随机参与率默认为false，每轮参与比例为1，因此这里就是所有客户端的序列
            self.send_models()  # 如果是第一轮，客户端把全局模型参数复制到本地

            if i % self.eval_gap == 0:  # eval_gap：评估的轮次间隔，默认为1，也就是每轮都评估一次
                print(f"\n-------------轮次: {i}-------------")
                print("\n评估个性化模型")
                self.evaluate()
                # 每次评估后更新图表
                self.update_plot()

            # 每个客户端进行本地训练
            for client in self.selected_clients:
                client.train()

            # 接收客户端上传的模型参数（不计算权重,只需要拿到客户端序号和模型即可）
            self.receive_models()

            # 对接收到的模型执行前向扩散，并记录潜在编码
            self.record_latent_encodings()

            # 训练扩散模型（用于噪声预测）
            self.train_diffusion_model()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, '本轮时间花费', '-' * 25, self.Budget[-1])

            # auto_break默认是false
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\n最佳精度.")
        print(max(self.rs_test_acc))

        print("\n每回合的平均时间成本.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

    # 条件扩散模型，用掩码作为条件信息
    class ConditionalDiffusionModel(nn.Module):
        def __init__(self, param_dim, hidden_dim):
            super().__init__()
            # 输入拼接 x (param_dim) + t (1) + mask (param_dim)，因此输入维度为 2*param_dim+1
            self.fc1 = nn.Linear(2 * param_dim + 1, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, param_dim)

        def forward(self, x, t, mask):
            # x: (B, param_dim)
            # t: (B, 1) 或 (B,)，统一转换为 (B, 1)
            # mask: (B, param_dim)；转换为浮点型并转移到 x 的设备
            t = t.view(-1, 1)
            mask = mask.float().to(x.device)
            input_tensor = torch.cat([x, t, mask], dim=1)
            h = torch.relu(self.fc1(input_tensor))
            h = torch.relu(self.fc2(h))
            out = self.fc3(h)
            return out

    def record_latent_encodings(self):
        """
        对每个上传的客户端模型参数 x0 (flatten后) 进行 T 步的前向扩散：
          x_t = sqrt(1 - beta[t]) * x_{t-1} + sqrt(beta[t]) * noise_t
        并将 (x_T, noise_T, ..., noise_1) 拼接成潜在编码 latent_code，同时保存对应的客户端关键参数掩码。
        保存格式：self.latent_encodings[cid] = {'latent_code': latent_code, 'mask': mask}
        """

        # 内部辅助函数，将模型中所有参数展平成一维向量
        def flatten_model(m):
            return torch.cat([p.data.view(-1) for p in m.parameters()])

        # 构造 beta 数组（线性从 1e-4 到 0.02，共 diff_steps 步）
        beta_start, beta_end = 1e-4, 0.02
        beta = torch.linspace(beta_start, beta_end, self.diff_steps, device=self.device)

        # 初始化 latent_encodings 字典，用于存储每个客户端的潜在编码及其关键参数掩码
        self.latent_encodings = {}

        # 遍历上传的客户端模型列表与对应的客户端 ID
        # 同时假设 self.uploaded_masks 已经通过 receive_models() 收到，并且与模型顺序一致
        for idx, (cid, model) in enumerate(zip(self.uploaded_ids, self.uploaded_models)):
            # 将当前客户端模型展平成一维向量 x0
            x0 = flatten_model(model).clone()  # (param_dim,)
            x_t = x0.clone()
            noise_list = []

            # 进行 diff_steps 步前向扩散
            for t in range(self.diff_steps):
                # 生成当前步的噪声 eps_t
                eps_t = torch.randn_like(x_t)
                # 根据扩散公式更新 x_t
                x_t = torch.sqrt(1 - beta[t]) * x_t + torch.sqrt(beta[t]) * eps_t
                # 记录当前步的噪声
                noise_list.append(eps_t)

            # 前向扩散结束后，x_t 即为 x_T
            x_T = x_t.clone()
            # 将噪声列表逆序排列，以便在反向扩散时按照从 T 开始的顺序使用
            reversed_noises = noise_list[::-1]
            # 拼接潜在编码：由 x_T 和各步噪声构成
            latent_code = torch.cat([x_T] + reversed_noises, dim=0)

            # 获取对应客户端上传的关键参数掩码（已展平）
            mask = self.uploaded_masks[idx]

            # 将潜在编码及掩码一起保存，便于后续扩散模型中作为条件输入
            self.latent_encodings[cid] = {'latent_code': latent_code, 'mask': mask}

            # 可选：调试信息
            # print(f"客户端 {cid} 的前向扩散完成，潜在编码和掩码已记录.")

    def train_diffusion_model(self):
        """
        训练 DDPM-like 多步扩散模型：在每个 batch 中，
          1) x0 = (B, param_dim)
          2) 随机采样 t in [1, T]
          3) 逐步从 x0 生成 x_t，期间记录最后一步的噪声 noise_t
          4) 扩散模型 (x_t, t, mask) -> 预测 noise_t
          5) 计算 MSE(noise_t, pred_noise) 损失，更新扩散模型
        其中 mask 为关键参数掩码（与 x0 同维度），作为条件输入。
        """
        # 如果还未上传模型，则直接返回
        if len(self.uploaded_models) == 0:
            print("尚无上传模型，无法训练扩散模型。")
            return

        # 辅助函数：展平模型参数
        def flatten_model(m):
            return torch.cat([p.data.view(-1) for p in m.parameters()])

        # 构造模型参数向量列表和对应的掩码列表
        model_vecs = []
        mask_vecs = []
        for idx, m in enumerate(self.uploaded_models):
            model_vecs.append(flatten_model(m))
            # 对应的掩码从 receive_models 中已经存储在 self.uploaded_masks 中
            mask_vecs.append(self.uploaded_masks[idx])
        # 堆叠成张量，形状分别为 (N, param_dim)
        model_params_tensor = torch.stack(model_vecs).to(self.device)
        mask_tensor = torch.stack(mask_vecs).to(self.device)

        n_samples, param_dim = model_params_tensor.shape

        # 如果扩散模型未初始化，则实例化条件扩散模型
        if self.diffusion_model is None:
            self.diffusion_model = self.ConditionalDiffusionModel(param_dim, self.hidden_dim).to(self.device)

        optimizer = optim.Adam(self.diffusion_model.parameters(), lr=self.diff_lr)
        loss_fn = nn.MSELoss()

        # 构造 beta，线性从 1e-4 到 0.02，共 diff_steps 步
        beta_start, beta_end = 1e-4, 0.02
        beta = torch.linspace(beta_start, beta_end, self.diff_steps, device=self.device)

        # 构建数据集，数据包含模型参数和对应的掩码
        dataset = torch.utils.data.TensorDataset(model_params_tensor, mask_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.diff_bs, shuffle=True)

        self.diffusion_model.train()
        for epoch in range(self.diff_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                # 从 batch 中提取 x0 和 mask，x0: (B, param_dim), mask: (B, param_dim)
                x0, mask = batch[0], batch[1]
                B = x0.size(0)

                # 随机为每个样本采样一个时间步 t，范围为 [1, diff_steps]
                t_array = torch.randint(1, self.diff_steps + 1, (B,), device=self.device)
                # 归一化时间步，范围 [0, 1]
                t_normalized = (t_array - 1).float() / float(self.diff_steps)

                # 对每个样本，利用多步前向扩散生成 x_t，并记录最后一次添加的噪声
                x_t = x0.clone()
                noise_t = torch.zeros_like(x_t)
                for idx_in_batch in range(B):
                    t_val = t_array[idx_in_batch]
                    cur_x = x0[idx_in_batch:idx_in_batch + 1, :].clone()
                    noise_ = None
                    for step_i in range(t_val):
                        eps = torch.randn_like(cur_x)
                        cur_x = torch.sqrt(1 - beta[step_i]) * cur_x + torch.sqrt(beta[step_i]) * eps
                        noise_ = eps  # 记录最后一次噪声
                    x_t[idx_in_batch] = cur_x
                    noise_t[idx_in_batch] = noise_.clone()

                # 预测噪声时，将 x_t、归一化时间 t_normalized 和对应的 mask 一同作为条件输入
                pred_noise = self.diffusion_model(x_t, t_normalized, mask)  # 输出形状 (B, param_dim)
                loss = loss_fn(pred_noise, noise_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * B

            avg_loss = epoch_loss / n_samples
            print(f"[扩散模型训练] epoch={epoch + 1}/{self.diff_epochs}, loss={avg_loss:.6f}")

    def get_personalized_parameters(self):
        """
        利用前向扩散过程中存储的潜在编码 (latent_code) 和对应的关键参数掩码 mask，
        结合扩散模型的噪声预测，采用反向扩散过程生成个性化模型参数。

        整体流程：
          1) 对于每个客户端，从 latent_encodings 中取出 latent_info，其中包含
             latent_code = (x_T, eps_T, eps_{T-1}, ..., eps_1) 和对应的 mask。
          2) 令当前状态 x_t 初始化为 x_T。
          3) 从 t = T 反推到 t = 1：对于每个时刻 t，
               - 利用扩散模型（条件输入 x_t、归一化时间 t 以及掩码 mask）预测噪声，
               - 根据公式： x_{t-1} = μₚ(x_t, t) - σ_t * eps_t，
                 其中 σ_t = sqrt(beta_t)（当 t>0 时，否则为 0），eps_t 为前向扩散时记录的噪声。
          4) 得到 x₀ 后，将其还原成模型各层参数构成个性化模型。

        关键公式：
          μₚ(x_t, t) = 1/sqrt(α_t) * [x_t - (β_t / sqrt(1 - ᾱ_t)) * pred_noise]
          其中 pred_noise = diffusion_model(x_t, t_normalized, mask)
        """
        if self.diffusion_model is None or not self.latent_encodings:
            print("扩散模型或潜在编码未就绪.")
            return

        # --- 设置扩散相关参数 ---
        beta_start, beta_end = 1e-4, 0.02
        beta = torch.linspace(beta_start, beta_end, self.diff_steps, device=self.device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        # 清空之前存储的个性化模型列表
        self.personalized_models = []

        # 定义辅助函数，计算均值 μₚ，加入 mask 作为条件输入
        def compute_mu(xt, t_idx, mask):
            """
            计算均值 μₚ(x_t, t) 的公式：
              μₚ(x_t, t) = 1/sqrt(α_t) * ( x_t - (β_t / sqrt(1 - ᾱ_t)) * pred_noise )
            其中 pred_noise 由扩散模型根据 (x_t, t_normalized, mask) 预测得到。
            """
            t_normalized = (t_idx / self.diff_steps) * torch.ones((1,), device=self.device)
            pred_noise = self.diffusion_model(xt, t_normalized, mask)
            a_t = alpha[t_idx]
            ab_t = alpha_bar[t_idx]
            sqrt_a_t = torch.sqrt(a_t)
            sqrt_one_minus_ab = torch.sqrt(1.0 - ab_t)
            beta_t = beta[t_idx]
            mu = (1.0 / sqrt_a_t) * (xt - (beta_t / sqrt_one_minus_ab) * pred_noise)
            return mu

        # 遍历每个客户端的上传信息
        for cid, template in zip(self.uploaded_ids, self.uploaded_models):
            # 从 latent_encodings 中取出该客户端的记录，包含 latent_code 和 mask
            latent_info = self.latent_encodings[cid]
            latent_code = latent_info['latent_code']
            mask = latent_info['mask']  # 展平后的掩码，形状与模型参数向量一致

            # 根据 latent_code 长度和扩散步数计算每个子向量的维度 param_dim
            param_dim = latent_code.size(0) // (self.diff_steps + 1)

            # 前 param_dim 个为 x_T
            x_T = latent_code[:param_dim].clone().view(1, -1)
            # 提取各步的噪声，noise_list[0] 对应 eps_T, noise_list[1] 对应 eps_{T-1}，依此类推
            noise_list = []
            for step_idx in range(self.diff_steps):
                start_idx = (step_idx + 1) * param_dim
                end_idx = (step_idx + 2) * param_dim
                eps_ = latent_code[start_idx:end_idx].clone().view(1, -1)
                noise_list.append(eps_)

            # 反向扩散过程，从 t = T 逐步还原到 t = 0
            cur_x = x_T
            # 同时将 mask 转换为 (1, param_dim)
            mask = mask.view(1, -1).float()
            for k in range(self.diff_steps):
                t_idx = self.diff_steps - 1 - k  # 当前时间步索引 t_idx ∈ [0, T-1]
                eps_k = noise_list[k]  # 对应的噪声 eps_t
                mu_t = compute_mu(cur_x, t_idx, mask)
                # 计算 σ_t，当 t_idx==0 时设为 0
                if t_idx == 0:
                    sigma_t = torch.tensor(0.0, device=self.device)
                else:
                    sigma_t = torch.sqrt((1 - alpha_bar[t_idx - 1]) / (1 - alpha_bar[t_idx]) * beta[t_idx])
                # 根据反向扩散公式
                prev_x = mu_t - sigma_t * eps_k
                cur_x = prev_x.clone()

            # cur_x 即为恢复得到的 x_0
            x_0 = cur_x.view(-1)

            # 将 x_0 还原为模型各层参数构成新的个性化模型
            new_model = copy.deepcopy(template)
            idx_ = 0
            for p in new_model.parameters():
                n_p = p.numel()
                p.data = x_0[idx_: idx_ + n_p].view(p.shape).clone()
                idx_ += n_p

            # 检查 NaN，若存在则回退到全局模型
            if any(torch.isnan(p.data).any() for p in new_model.parameters()):
                print(f"警告：客户端 {cid} 用扩散模型反向生成后出现NaN, fallback 用 global_model.")
                new_model = copy.deepcopy(self.global_model)
            self.personalized_models.append(new_model)

    def send_models(self):
        if self.epoch != 0:
            self.get_personalized_parameters()
            for client in self.clients:
                start_time = time.time()
                if client.id in self.uploaded_ids:
                    idx = self.uploaded_ids.index(client.id)
                    client.set_parameters(self.personalized_models[idx])
                else:
                    client.set_parameters(self.global_model)
                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
        else:
            for client in self.clients:
                start_time = time.time()
                client.set_parameters(self.global_model)
                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0
        self.uploaded_ids = [c.id for c in self.selected_clients]
        self.uploaded_models = [c.model for c in self.selected_clients]
        self.uploaded_masks = []
        # 遍历每个客户端，接收其计算好的 local_mask
        for client in self.selected_clients:
            if client.local_mask is not None:
                # 将每层的掩码展平后拼接成一个长向量
                flat_mask = torch.cat([m.view(-1) for m in client.local_mask])
            else:
                # 如果客户端未计算 local_mask，则构造一个与模型参数维度相同的全零向量
                flat_mask = torch.zeros(sum(p.numel() for p in client.model.parameters()), device=self.device)
            self.uploaded_masks.append(flat_mask)

    def update_plot(self):
        """
        实时更新图表：绘制测试准确率、训练损失和测试AUC随全局轮次变化的图形，
        并调用 plt.pause() 使图形窗口更新。
        """
        # 确保至少有一项评估数据
        if len(self.rs_test_acc) == 0 and len(self.rs_train_loss) == 0 and len(self.rs_test_auc) == 0:
            return

        # 以 rs_test_acc 的长度作为 x 轴长度
        rounds = list(range(0, len(self.rs_test_acc) * self.eval_gap, self.eval_gap))
        # 若 rs_test_acc 数据为空，则设置 rounds 为空列表
        if len(rounds) == 0:
            rounds = []

        plt.clf()  # 清空当前图形

        # 绘制测试准确率，如果数据存在
        if len(self.rs_test_acc) > 0:
            plt.subplot(1, 3, 1)
            plt.plot(rounds, self.rs_test_acc, marker='o', color='blue', label='Test Accuracy')
            plt.xlabel('Global Round')
            plt.ylabel('Test Accuracy')
            plt.title('Test Accuracy vs. Global Rounds')
            plt.legend()
            plt.grid(True)

        # 绘制训练损失，如果数据存在
        if len(self.rs_train_loss) > 0:
            plt.subplot(1, 3, 2)
            plt.plot(rounds, self.rs_train_loss, marker='o', color='red', label='Train Loss')
            plt.xlabel('Global Round')
            plt.ylabel('Train Loss')
            plt.title('Train Loss vs. Global Rounds')
            plt.legend()
            plt.grid(True)

        # 绘制测试AUC，如果数据存在
        if len(self.rs_test_auc) > 0:
            plt.subplot(1, 3, 3)
            # 以 rs_test_auc 的长度为准
            auc_rounds = list(range(0, len(self.rs_test_auc) * self.eval_gap, self.eval_gap))
            plt.plot(auc_rounds, self.rs_test_auc, marker='o', color='green', label='Test AUC')
            plt.xlabel('Global Round')
            plt.ylabel('Test AUC')
            plt.title('Test AUC vs. Global Rounds')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.draw()  # 刷新图形
        plt.pause(0.1)  # 暂停0.1秒以显示更新
