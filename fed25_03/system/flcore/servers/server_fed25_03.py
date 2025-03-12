import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from flcore.clients.client_fed25_03 import Client_fed03
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data


class Server_fed03(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 初始化动态掩码生成的超参数
        self.alpha_min = float(args.alpha_min)
        self.alpha_max = float(args.alpha_max)
        self.alpha_k = float(args.alpha_k)

        # 初始化扩散模型训练相关超参数
        self.diff_steps = args.diff_steps
        self.noise_schedule = args.noise_schedule
        self.diff_lr = args.diff_lr
        self.diff_epochs = args.diff_epochs
        self.diff_bs = args.diff_bs
        self.guidance_weight = args.guidance_weight
        self.denoise_steps = args.denoise_steps
        self.hidden_dim = args.hidden_dim
        self.latent_dim = args.latent_dim
        self.personalized_models = []

        # 为扩散模型和潜在编码预留占位符（后续训练中实例化）
        self.diffusion_model = None
        self.latent_encodings = {}  # 存储每个客户端的潜在编码γ

        # 随机选择慢客户端（但现在没开）
        self.set_slow_clients()  # 根据train_slow_rate标记哪些客户端在训练阶段是慢的，根据send_slow_rate标记哪些在发送阶段是慢的
        self.set_clients(Client_fed03)  # 设置客户端，传入 clientCAC 类

        print(f"\n参与率 / 总客户端数: {self.join_ratio} / {self.num_clients}")
        print("完成创建服务端和客户端")

        self.Budget = []  # 初始化空的 Budget 列表，用于存储时间开销

        # To be consistent with the existing pipeline interface. Maintaining an epoch counter.
        self.epoch = -1  # 设置epoch 为 -1，这和self.local_epoch不是同一个

    def train(self):
        for i in range(self.global_rounds + 1):
            self.epoch = i  # 更新当前的 epoch 值
            s_t = time.time()  # 记录当前时间，计算训练时间
            self.selected_clients = self.select_clients()  # 这里随机参与率默认为false，每轮参与比例为1，因此这里就是所有客户端的序列
            self.send_models()  # 如果是第一轮，客户端把全局模型参数复制到本地

            # ---- 在评估前做最后一次清理 ----
            self.clean_client_models()

            if i % self.eval_gap == 0:  # eval_gap：评估的轮次间隔，默认为1，也就是每轮都评估一次
                print(f"\n-------------轮次: {i}-------------")
                print("\n评估个性化模型")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # 接收客户端上传的模型参数（不计算权重）
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
            self.fc1 = nn.Linear(param_dim + 1, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, param_dim)

        def forward(self, x, t):
            # t 为归一化的时间步，形状 (B,)，扩展为 (B,1) 后与 x 拼接
            t = t.view(-1, 1)
            x_in = torch.cat([x, t], dim=1)
            h = torch.relu(self.fc1(x_in))
            h = torch.relu(self.fc2(h))
            out = self.fc3(h)
            return out

    def forward_diffusion(self, x0):
        """
        对输入的模型参数向量 x0 进行完整的前向扩散，
        返回最终加噪后的 x_T 以及潜在编码 gamma（由 x_T 与各步噪声拼接而成）。
        """
        beta_start = 1e-4
        beta_end = 0.02
        # 生成 diff_steps 个噪声调度参数 β
        beta = torch.linspace(beta_start, beta_end, self.diff_steps, device=self.device)  # shape: (diff_steps,)
        # 计算 α_t = ∏_{j=1}^t (1-β_j)
        alpha = torch.cumprod(1 - beta, dim=0)  # shape: (diff_steps,)
        noises = []
        x_t = x0
        # 模拟前向扩散，从 t=0 到 t=T（diff_steps-1）
        for t in range(self.diff_steps):
            noise = torch.randn_like(x0)
            x_t = torch.sqrt(1 - beta[t]) * x_t + torch.sqrt(beta[t]) * noise
            noises.append(noise)
        latent_code = torch.cat([x_t] + noises[::-1])
        return x_t, latent_code

    def record_latent_encodings(self):
        """
        对每个上传的客户端模型参数 x0 (flatten后) 进行T步的前向扩散：
           x_{t} = sqrt(1 - beta[t]) * x_{t-1} + sqrt(beta[t]) * noise_t
        并将 ( x_T, noise_T, ..., noise_1 ) 拼成 latent_code 存到 self.latent_encodings[cid].
        """

        def flatten_model(m):
            return torch.cat([p.data.view(-1) for p in m.parameters()])

        # 构造beta数组(例如线性从1e-4到0.02，共diff_steps步)
        beta_start, beta_end = 1e-4, 0.02
        beta = torch.linspace(beta_start, beta_end, self.diff_steps, device=self.device)  # (T,)

        self.latent_encodings = {}
        for i, (cid, model) in enumerate(zip(self.uploaded_ids, self.uploaded_models)):
            x0 = flatten_model(model).clone()  # (param_dim,)
            x_t = x0.clone()
            noise_list = []

            # 多步更新
            for t in range(self.diff_steps):
                # 生成本步噪声
                eps_t = torch.randn_like(x_t)
                # x_{t} = sqrt(1-beta[t]) * x_{t-1} + sqrt(beta[t]) * eps_t
                x_t = torch.sqrt(1 - beta[t]) * x_t + torch.sqrt(beta[t]) * eps_t
                noise_list.append(eps_t)

            # 现在 x_t 就是 x_T
            # 我们把 x_T 拼上 noise_{T}, noise_{T-1}, ..., noise_1
            # 注意：论文/示例中是 reverse，故可以 noise_list[::-1]
            # latent_code 大小 = (T+1)*param_dim
            x_T = x_t.clone()
            reversed_noises = noise_list[::-1]
            latent_code = torch.cat([x_T] + reversed_noises, dim=0)

            self.latent_encodings[cid] = latent_code

            print(f"客户端 {cid} 的多步前向扩散完成.")

    def train_diffusion_model(self):
        """
        训练DDPM-like 的多步: 在每个batch中:
          1) x0 = (B, param_dim)
          2) 随机采样 t in [1..T]
          3) 逐步地 x_1... x_t, 期间最后一次加噪 noise_t
          4) 扩散模型( x_t, t ) -> 预测 noise_t
          5) MSE(noise_t, pred_noise)
        """
        if len(self.uploaded_models) == 0:
            print("尚无上传模型，无法训练扩散模型。")
            return

        # flatten
        def flatten_model(m):
            return torch.cat([p.data.view(-1) for p in m.parameters()])

        model_vecs = []
        for m in self.uploaded_models:
            model_vecs.append(flatten_model(m))
        model_params_tensor = torch.stack(model_vecs).to(self.device)  # shape (N, param_dim)
        n_samples, param_dim = model_params_tensor.shape

        if self.diffusion_model is None:
            self.diffusion_model = self.SimpleDiffusionModel(param_dim, self.hidden_dim).to(self.device)

        optimizer = optim.Adam(self.diffusion_model.parameters(), lr=self.diff_lr)
        loss_fn = nn.MSELoss()

        # 构造 beta
        beta_start, beta_end = 1e-4, 0.02
        beta = torch.linspace(beta_start, beta_end, self.diff_steps, device=self.device)  # (T,)

        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(model_params_tensor)
        dataloader = DataLoader(dataset, batch_size=self.diff_bs, shuffle=True)

        self.diffusion_model.train()

        for epoch in range(self.diff_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                x0 = batch[0]  # (B, param_dim)
                B = x0.size(0)

                # 1) 随机 t in [1..T], 并把它归一化
                t_array = torch.randint(1, self.diff_steps + 1, (B,), device=self.device)  # t in [1..T]
                # 之后要传给网络作为embedding/feature
                t_normalized = (t_array - 1).float() / float(self.diff_steps)  # (B,), 让 t-1 变成[0..T-1]

                # 2) 多步地从x0走到x_t
                x_t = x0.clone()
                noise_t = torch.zeros_like(x_t)  # 用来记录“最后一步 noise”
                for idx_in_batch in range(B):
                    t_val = t_array[idx_in_batch]  # 该sample的t
                    # 逐步扩散
                    cur_x = x0[idx_in_batch:idx_in_batch + 1, :].clone()  # shape (1, param_dim)
                    noise_ = None
                    for step_i in range(t_val):
                        eps = torch.randn_like(cur_x)
                        cur_x = torch.sqrt(1 - beta[step_i]) * cur_x + torch.sqrt(beta[step_i]) * eps
                        noise_ = eps  # 最后一步的eps
                    # 把结果放回 x_t
                    x_t[idx_in_batch] = cur_x
                    noise_t[idx_in_batch] = noise_.clone()

                # 3) pred_noise = model( x_t, t_normalized )
                pred_noise = self.diffusion_model(x_t, t_normalized)  # (B, param_dim)

                # 4) 计算 MSE
                loss = loss_fn(pred_noise, noise_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * B

            avg_loss = epoch_loss / n_samples
            print(f"[Diffusion Train] epoch={epoch + 1}/{self.diff_epochs}, loss={avg_loss:.6f}")

    def get_personalized_parameters(self):
        """
        利用多步前向时存好的: latent_code = ( x_T, eps_T, eps_{T-1}, ..., eps_1 )
        逐步做:
           x_{k-1} = ( x_k - sqrt(beta[k-1])*eps_k ) / sqrt(1 - beta[k-1])
        重建 x_0 -> 填充回模型
        """
        if self.diffusion_model is None or not self.latent_encodings:
            print("扩散模型或潜在编码未就绪.")
            return

        # 准备 beta
        beta_start, beta_end = 1e-4, 0.02
        beta = torch.linspace(beta_start, beta_end, self.diff_steps, device=self.device)  # (T,)

        self.personalized_models = []

        for i, cid in enumerate(self.uploaded_ids):
            latent_code = self.latent_encodings[cid]  # shape: ( (T+1)*param_dim, )
            param_dim = latent_code.size(0) // (self.diff_steps + 1)
            # 前 param_dim 即 x_T
            x_T = latent_code[:param_dim].clone()

            # 依次取 eps_T, eps_{T-1},...,eps_1
            # 逆序存储
            noise_list = []
            for step_idx in range(self.diff_steps):
                start_ = (step_idx + 1) * param_dim
                end_ = (step_idx + 2) * param_dim
                eps_ = latent_code[start_: end_]
                noise_list.append(eps_)
            # noise_list[0] = eps_T, noise_list[1] = eps_{T-1} ...
            # 但我们要按照 T, T-1, ... 1 的顺序反推
            # 也可以直接 noise_list 里看索引

            # 现在多步反向
            cur_x = x_T.view(-1)  # param_dim
            for k in range(self.diff_steps):
                # 第 k 步反向对应 x_{T-k} -> x_{T-k-1}
                # 这里 index = k,  t = T-k
                # 在正向时: x_{t} = sqrt(1-beta[t-1])* x_{t-1} + sqrt(beta[t-1])*eps_t
                # => x_{t-1} = ( x_t - sqrt(beta[t-1])* eps_t ) / sqrt(1-beta[t-1])
                eps_k = noise_list[k]  # = eps_{T-k}, k=0->eps_T
                # beta的下标 = T-1-k
                beta_idx = self.diff_steps - 1 - k
                b = beta[beta_idx]
                denominator = torch.sqrt(1. - b)
                # 反推:
                prev_x = (cur_x - torch.sqrt(b) * eps_k) / (denominator + 1e-12)
                cur_x = prev_x.clone()

            # 最终 cur_x 就是 x_0
            x_0 = cur_x

            # 填充回模型
            template = self.uploaded_models[i]
            new_model = copy.deepcopy(template)
            idx_ = 0
            for p in new_model.parameters():
                n_p = p.numel()
                p.data = x_0[idx_: idx_ + n_p].view(p.shape).clone()
                idx_ += n_p

            # 检查NaN
            has_nan = any(torch.isnan(p.data).any() for p in new_model.parameters())
            if has_nan:
                print(f"警告：客户端 {cid} 反向去噪后出现NaN, fallback 用 global_model.")
                new_model = copy.deepcopy(self.global_model)
            else:
                print(f"客户端 {cid} 个性化参数生成完成(多步).")

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
