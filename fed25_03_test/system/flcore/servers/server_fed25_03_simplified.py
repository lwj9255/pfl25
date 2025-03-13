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

    # 定义一个简单的扩散模型（MLP），将模型参数向量和归一化的时间步 t 作为输入，输出与参数向量同维度的预测噪声
    class SimpleDiffusionModel(nn.Module):
        def __init__(self, param_dim, hidden_dim):
            super().__init__()
            # 三层全连接网络：输入为参数向量与时间步拼接
            self.fc1 = nn.Linear(param_dim + 1, hidden_dim)  # 第一层全连接层的输入维度是param_dim + 1，因为要拼接时间步 t
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, param_dim)

        def forward(self, x, t):
            # 将时间步 t 转换成形状 (B, 1) 并与 x 拼接，B 表示当前批次（batch）的样本数量
            t = t.view(-1, 1)
            x_in = torch.cat([x, t], dim=1)  # 拼接之后，形状会是 (B, param_dim + 1)
            h = torch.relu(self.fc1(x_in))
            h = torch.relu(self.fc2(h))
            out = self.fc3(h)
            return out

    def record_latent_encodings(self):
        """
        对每个上传的客户端模型参数 x0 (flatten后) 进行T步的前向扩散：
           x_{t} = sqrt(1 - beta[t]) * x_{t-1} + sqrt(beta[t]) * noise_t
        并将 ( x_T, noise_T, ..., noise_1 ) 拼成 潜在编码latent_code 存到 self.latent_encodings[cid].
        """

        # 定义一个内部辅助函数 flatten_model，用于将传入模型中所有参数展平成一个一维向量。
        # 这里利用列表推导，对模型中每个参数 p 调用 p.data.view(-1) 将其变为一维张量，
        # 然后通过 torch.cat 将所有这些一维张量连接起来，形成一个长向量，代表整个模型的参数。
        def flatten_model(m):
            return torch.cat([p.data.view(-1) for p in m.parameters()])

        # 构造beta数组(线性从1e-4到0.02，共diff_steps步)
        beta_start, beta_end = 1e-4, 0.02
        beta = torch.linspace(beta_start, beta_end, self.diff_steps, device=self.device)

        # 初始化 self.latent_encodings 字典，用来存储每个客户端的潜在编码。
        # 字典的键是客户端的id，值则是对应的 latent_code。
        self.latent_encodings = {}

        # 遍历上传的客户端模型列表 self.uploaded_models 与对应的客户端 id 列表 self.uploaded_ids，
        for cid, model in zip(self.uploaded_ids, self.uploaded_models):
            # 调用 flatten_model 函数，将当前客户端上传的模型 model 中所有参数展平成一个一维张量 x0。
            x0 = flatten_model(model).clone()  # (param_dim,)
            x_t = x0.clone()
            # 初始化 noise_list 用来存储每步添加的噪声
            noise_list = []

            # 多步更新
            for t in range(self.diff_steps):
                # 生成本步噪声
                eps_t = torch.randn_like(x_t)  # 生成与 x_t 同形状的标准正态噪声 eps_t。
                # 按照扩散公式计算x_{t} = sqrt(1-beta[t]) * x_{t-1} + sqrt(beta[t]) * eps_t
                x_t = torch.sqrt(1 - beta[t]) * x_t + torch.sqrt(beta[t]) * eps_t
                # 将本步产生的噪声 eps_t 存入 noise_list 中
                noise_list.append(eps_t)

            # 前向传播diff_steps轮后， x_t 就是 x_T
            # 把 x_T 拼上 noise_{T}, noise_{T-1}, ..., noise_1
            # 将 noise_list 逆序排列（因为反演时需要逆序使用噪声，即从 noise_T 开始），故可以 noise_list[::-1]
            # latent_code 大小 = (T+1)*param_dim，是一个展平的向量
            x_T = x_t.clone()
            reversed_noises = noise_list[::-1]
            latent_code = torch.cat([x_T] + reversed_noises, dim=0)

            # 将生成的 latent_code 存入字典中，键为当前客户端 id
            self.latent_encodings[cid] = latent_code

            # print(f"客户端 {cid} 的多步前向扩散完成.")

    def train_diffusion_model(self):
        """
        训练DDPM-like 的多步: 在每个batch中:
          1) x0 = (B, param_dim)
          2) 随机采样 t in [1..T]
          3) 逐步地 x_1... x_t, 期间最后一次加噪 noise_t
          4) 扩散模型( x_t, t ) -> 预测 noise_t
          5) MSE(noise_t, pred_noise)
        """
        # 检查，但实际上不会触发的
        if len(self.uploaded_models) == 0:
            print("尚无上传模型，无法训练扩散模型。")
            return

        # 同前向扩散，将模型参数展平成一维向量
        def flatten_model(m):
            return torch.cat([p.data.view(-1) for p in m.parameters()])

        # 遍历所有上传的模型，将展平后的向量存入列表 model_vecs
        model_vecs = []
        for m in self.uploaded_models:
            model_vecs.append(flatten_model(m))
        # 利用 torch.stack 将所有模型向量堆叠成一个张量，形状为 (N, param_dim)。
        model_params_tensor = torch.stack(model_vecs).to(self.device)
        n_samples, param_dim = model_params_tensor.shape

        # 如果扩散模型还未初始化，则实例化一个简单的扩散模型（一个三层全连接网络）
        # 输入维度为 param_dim，加上一个时间步特征，隐藏层大小为 self.hidden_dim。
        if self.diffusion_model is None:
            self.diffusion_model = self.SimpleDiffusionModel(param_dim, self.hidden_dim).to(self.device)

        # 采用 Adam 优化器优化扩散模型参数，学习率为 self.diff_lr；损失函数为均方误差（MSE）。
        optimizer = optim.Adam(self.diffusion_model.parameters(), lr=self.diff_lr)
        loss_fn = nn.MSELoss()

        # 构造 beta，同前向扩散过程
        beta_start, beta_end = 1e-4, 0.02
        beta = torch.linspace(beta_start, beta_end, self.diff_steps, device=self.device)

        from torch.utils.data import DataLoader, TensorDataset
        # 将所有展平后的模型参数张量作为数据集
        # 利用 DataLoader 按批次（batch size 为 self.diff_bs）加载数据
        # 随机打乱顺序以增加训练多样性。
        dataset = TensorDataset(model_params_tensor)
        dataloader = DataLoader(dataset, batch_size=self.diff_bs, shuffle=True)

        # 将扩散模型切换到训练模式
        self.diffusion_model.train()

        for epoch in range(self.diff_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                # 从当前 batch 中取出模型参数向量 x0，形状为 (B, param_dim)
                x0 = batch[0]
                # B 表示当前 batch 的样本数。
                B = x0.size(0)

                # 1) 为每个样本随机选择一个时间步 t（范围 1 到 T）。
                t_array = torch.randint(1, self.diff_steps + 1, (B,), device=self.device)
                # 将 t 转换为浮点数，并归一化到 [0, 1]（用于作为扩散模型的输入特征）。
                # 归一化后的时间步输入范围固定，有助于模型在训练时保持数值稳定，避免因时间步数值跨度较大而影响梯度更新
                t_normalized = (t_array - 1).float() / float(self.diff_steps)  # (B,), 让 t-1 变成[0..T-1]

                # 2) 多步地从x0走到x_t
                x_t = x0.clone()  # (B, param_dim)
                noise_t = torch.zeros_like(x_t)  # 用来记录“最后一步 noise”，(B, param_dim)
                for idx_in_batch in range(B):  # 对于批次中每个样本
                    t_val = t_array[idx_in_batch]  # 取出该样本随机采样到的 t 值 t_val。
                    # 逐步扩散
                    # 从 x0 中取出下标从 idx_in_batch 开始到 idx_in_batch+1 之前的部分拷贝为 cur_x
                    cur_x = x0[idx_in_batch:idx_in_batch + 1, :].clone()  # shape (1, param_dim)
                    noise_ = None
                    # 在 t_val 个步骤内依次执行前向扩散，每步添加噪声 eps 并更新 cur_x。
                    for step_i in range(t_val):
                        eps = torch.randn_like(cur_x)
                        cur_x = torch.sqrt(1 - beta[step_i]) * cur_x + torch.sqrt(beta[step_i]) * eps
                        noise_ = eps  # 将最后一次添加的噪声（即在第 t_val 步中的 eps）记录为 noise_。
                    # 将扩散后的结果 cur_x 存入 x_t，对应位置的 noise_ 存入 noise_t。
                    x_t[idx_in_batch] = cur_x
                    noise_t[idx_in_batch] = noise_.clone()

                # 3) 将批次中扩散后的参数 x_t 与归一化的时间步 t_normalized 一起输入扩散模型
                # 模型输出预测的噪声 pred_noise
                pred_noise = self.diffusion_model(x_t, t_normalized)  # (B, param_dim)

                # 4) 计算 MSE损失
                loss = loss_fn(pred_noise, noise_t)  # 计算预测噪声 pred_noise 与真实最后一步噪声 noise_t 之间的均方误差损失
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * B

            avg_loss = epoch_loss / n_samples
            print(f"[扩散模型训练] epoch={epoch + 1}/{self.diff_epochs}, loss={avg_loss:.6f}")

    def get_personalized_parameters(self):
        """
        利用前向扩散过程中存储的潜在编码 (latent_code)，结合扩散模型的噪声预测，
        采用反向扩散过程生成个性化模型参数。

        整体流程：
          1) 对于每个客户端，取出 latent_code = (x_T, eps_T, eps_{T-1}, ..., eps_1)，
             其中 x_T 为前向扩散最后一步的结果，后续依次为每一步添加的噪声（逆序存储）。
          2) 令当前状态 x_t 初始化为 x_T。
          3) 从 t = T 反推到 t = 1：对于每个时刻 t，
               - 利用已训练的扩散模型预测噪声，从而计算均值 μₚ（记作 mu_t)；
               - 根据公式： x_{t-1} = μₚ(x_t, t) - sigma_t * eps_t，
                 其中 sigma_t 取为 sqrt(beta_t)，eps_t 为前向扩散时保存的噪声。
          4) 最终得到 x₀，将 x₀还原回模型各层参数，构成个性化模型。
        """
        if self.diffusion_model is None or not self.latent_encodings:
            print("扩散模型或潜在编码未就绪.")
            return

        # --- 准备反向扩散所需的超参数 ---
        # 设置 beta 参数线性变化范围：从 1e-4 到 0.02，共 self.diff_steps 步
        beta_start, beta_end = 1e-4, 0.02
        beta = torch.linspace(beta_start, beta_end, self.diff_steps, device=self.device)  # beta: shape (T,)
        # 定义 alpha_t = 1 - beta_t，对于每一步
        alpha = 1.0 - beta  # shape (T,)
        # 计算 ᾱ_t，即从 1 到 t 所有 alpha 的乘积，代表累计保留比例
        alpha_bar = torch.cumprod(alpha, dim=0)  # shape (T,)

        # 清空之前存储的个性化模型列表
        self.personalized_models = []

        # --- 定义一个辅助函数，用于计算当前状态 x_t 在时刻 t 的均值 μₚ ---
        def compute_mu(xt, t_idx):
            """
            计算均值 μₚ(x_t, t) 的公式：
              μₚ(x_t, t) = 1/sqrt(alpha_t) * ( x_t - (beta_t / sqrt(1 - alpha_bar[t])) * pred_noise )
            其中：
              - xt: 当前时刻的状态 (形状为 (1, param_dim))
              - t_idx: 当前时间步的索引（在 [0, T-1] 内）
              - pred_noise: 由扩散模型根据 xt 和归一化时间 t 预测的噪声
            """
            # 将 t_idx 归一化到 [0, 1]，构成扩散模型的时间输入，此处假设批次大小为 1
            t_normalized = (t_idx / self.diff_steps) * torch.ones((1,), device=self.device)

            # 利用扩散模型预测当前时刻的噪声，pred_noise 的形状与 xt 相同
            pred_noise = self.diffusion_model(xt, t_normalized)
            # 提取当前时间步的 alpha 和累积 alpha_bar
            a_t = alpha[t_idx]  # 当前时刻的 alpha_t
            ab_t = alpha_bar[t_idx]  # 当前时刻的 alpha_bar_t
            sqrt_a_t = torch.sqrt(a_t)  # 计算 sqrt(alpha_t)
            sqrt_one_minus_ab = torch.sqrt(1.0 - ab_t)  # 计算 sqrt(1 - alpha_bar_t)
            beta_t = beta[t_idx]  # 当前时刻的 beta_t
            # 计算均值 μₚ，分两部分：1/sqrt(alpha_t) 和 [x_t - (beta_t / sqrt(1 - alpha_bar_t)) * pred_noise]
            coeff_1 = 1.0 / sqrt_a_t
            coeff_2 = beta_t / sqrt_one_minus_ab
            mu = coeff_1 * (xt - coeff_2 * pred_noise)  # 得到预测均值，形状 (1, param_dim)
            return mu

        # -------------------------------------------------------------------------
        # 开始为每个客户端生成个性化模型
        for cid, template in zip(self.uploaded_ids, self.uploaded_models):
            # 从字典中取出当前客户端的 latent_code，形状为 ((T+1)*param_dim, )
            latent_code = self.latent_encodings[cid]
            # 计算每个子向量的维度 param_dim：latent_code 总长度除以 (T+1)
            param_dim = latent_code.size(0) // (self.diff_steps + 1)

            # 前 param_dim 个即 x_T
            x_T = latent_code[:param_dim].clone().view(1, -1)  # shape (1, param_dim)
            # 后续依次取 eps_T, eps_{T-1},...,eps_1
            # 初始化列表，用于存放各步添加的噪声（逆序存储）
            noise_list = []
            for step_idx in range(self.diff_steps):
                # 每一步的噪声占用 param_dim 个元素，起始索引为 (step_idx+1)*param_dim，结束索引为 (step_idx+2)*param_dim
                start_ = (step_idx + 1) * param_dim
                end_ = (step_idx + 2) * param_dim
                eps_ = latent_code[start_: end_].clone().view(1, -1)  # 保持 (1, param_dim)
                noise_list.append(eps_)
            # 注意：noise_list[0] 对应 eps_T，noise_list[1] 对应 eps_{T-1}，依此类推

            # --- 反向扩散过程，从 t = T 逐步还原到 t = 0 ---
            # 初始状态设为 x_T（记作 cur_x），形状为 (1, param_dim)
            cur_x = x_T

            for k in range(self.diff_steps):
                # 计算当前反向步骤对应的时间步索引 t_idx：
                # 当 k=0 时，t_idx = T-1（即最后一步）；当 k=1 时，t_idx = T-2，以此类推
                t_idx = self.diff_steps - 1 - k  # 在 PyTorch 张量中, [0..T-1]
                # 取出对应步骤保存的噪声 eps_t，注意 noise_list[k] 对应的是 eps_{T-k}
                eps_k = noise_list[k]  # = eps_{T-k}

                # --- 使用扩散模型预测当前状态 x_t 的均值 μₚ ---
                mu_t = compute_mu(cur_x, t_idx)  # 预测得到均值，形状为 (1, param_dim)

                # --- 计算 sigma_t ---
                # 根据简化公式，直接设 sigma_t = sqrt(beta[t_idx])
                if t_idx == 0:
                    sigma_t = torch.tensor(0.0, device=self.device)
                else:
                    sigma_t = torch.sqrt((1 - alpha_bar[t_idx - 1]) / (1 - alpha_bar[t_idx]) * beta[t_idx])

                # --- 反向还原公式 ---
                # 根据公式： x_{t-1} = μₚ(x_t, t) - sigma_t * eps_t
                # 这里用存储的 eps_t 替代随机采样噪声
                prev_x = mu_t - sigma_t * eps_k  # 得到反推一步后的状态 x_{t-1}，形状 (1, param_dim)

                # 更新当前状态，将 prev_x 作为下一次反推的输入
                cur_x = prev_x.clone()

            # 最终经过反向扩散后，cur_x 就是还原出的 x₀，即个性化模型的参数向量，形状为 (1, param_dim)
            x_0 = cur_x.view(-1)  # 展平回 (param_dim,)

            # 填充回模型
            new_model = copy.deepcopy(template)
            idx_ = 0
            for p in new_model.parameters():
                n_p = p.numel()
                p.data = x_0[idx_: idx_ + n_p].view(p.shape).clone()
                idx_ += n_p

            # 检查 NaN
            has_nan = any(torch.isnan(p.data).any() for p in new_model.parameters())
            if has_nan:
                print(f"警告：客户端 {cid} 用扩散模型反向生成后出现NaN, fallback 用 global_model.")
                new_model = copy.deepcopy(self.global_model)
            # else:
            #     print(f"客户端 {cid} 个性化参数生成完成 (多步 + 扩散模型).")

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

    def clean_client_models(self):
        has_fallback = False
        for client in self.clients:
            if any(torch.isnan(p.data).any() for p in client.model.parameters()):
                print(f"警告：客户端 {client.id} 模型含 NaN，回退至全局模型")
                client.set_parameters(self.global_model)
                has_fallback = True
        if has_fallback:
            print("已处理所有含 NaN 的客户端模型")

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
