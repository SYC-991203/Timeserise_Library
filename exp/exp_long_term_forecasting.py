from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.visual import *
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import re
import csv
warnings.filterwarnings('ignore')

class MoE_MSE_Loss(nn.Module):
    def __init__(self, model, lambda_moe=0.01, confidence_threshold=2.0):
        super(MoE_MSE_Loss, self).__init__()
        self.base_criterion = nn.MSELoss()
        self.model = model
        self.lambda_moe = lambda_moe
        
        # 阈值设定：
        # Logits 差值 > 2.0  => 概率偏向 > 88%  => 视为"强先验"，跳过均衡 Loss
        # Logits 差值 < 2.0  => 概率偏向 < 88%  => 视为"不确定"，施加均衡 Loss
        self.confidence_threshold = confidence_threshold

    def _compute_adaptive_moe_loss(self, routing_weights_dict, target_device):
        total_moe_loss = torch.tensor(0.0, device=target_device)
        active_groups = 0

        # 处理 DataParallel: 获取原始模型引用以访问 routers
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        
        for group_name, weights in routing_weights_dict.items():
            weights = weights.to(target_device)
            
            # === 1. 捕获 LLM 引导的 Logits ===
            # 从模型中获取对应组的 Router
            if group_name in model_ref.group_moe_routers:
                router = model_ref.group_moe_routers[group_name]
                
                # 获取可训练参数 guidance_logits: [1, 1, 2]
                # detach() 很重要！我们只用它做判断条件，不希望 Loss 反向传导去刻意缩小 Logits 差距
                logits = router.guidance_logits.detach().to(target_device)
                
                # 计算置信度 (Confidence): 两个专家 Logits 的绝对差值
                # logits[..., 0] - logits[..., 1]
                diff = torch.abs(logits[..., 0] - logits[..., 1]).mean()
                
                # === 2. 自适应判断 ===
                if diff > self.confidence_threshold:
                    # Case A: 强先验 (LLM 很自信 or 训练后模型变自信了)
                    # 允许坍塌，跳过均衡损失
                    continue 
            
            # === 3. 施加均衡损失 (只针对不确定组) ===
            # Case B: 弱先验/中立组
            # 计算 Load Balancing Loss (MSE of Expert Load)
            expert_load = weights.reshape(-1, weights.shape[-1]).mean(dim=0)
            # 目标是均匀分布 (1/N)，最小化方差
            load_loss = (expert_load ** 2).sum() * weights.shape[-1]
            
            total_moe_loss += load_loss
            active_groups += 1
            
        # 避免除以 0
        return total_moe_loss / (active_groups + 1e-6)

    def forward(self, pred, target):
        # 1. 主任务损失
        mse_loss = self.base_criterion(pred, target)
        
        # 2. 获取权重
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        routing_weights = getattr(model_ref, 'attention_results', {}).get('group_routing_weights', {})
        
        # 3. 计算自适应 MoE Loss
        aux_loss = self._compute_adaptive_moe_loss(routing_weights, target_device=pred.device)
        
        # 4. 总损失
        total_loss = mse_loss + self.lambda_moe * aux_loss
        print(f"MSE: {mse_loss.item():.6f} | MoE Aux: {aux_loss.item():.6f}")
        return total_loss
class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'>>> Total Params: {total_params:,}')
        print(f'>>> Trainable Params: {trainable_params:,}')
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
            # 如果参数指定了 MoE_MSE，则使用我们自定义的 Loss
            if self.args.loss == 'MoE_MSE':
                print(">>> Using LLM-Guided MoE Loss (MSE + Load Balancing) <<<")
                # 将 self.model 传给 Loss，实现状态共享
                return MoE_MSE_Loss(self.model, lambda_moe=0.01)
            
            # 否则使用默认的 MSE 或 L1
            criterion = nn.MSELoss()
            if self.args.loss == 'MAE':
                criterion = nn.L1Loss()
            return criterion

    def vali(self, vali_data, vali_loader, criterion):
        if self.args.model == 'PromptCast':
            print("PromptCast 不支持验证，执行推理即可。")
            return 0
        total_loss = []
        self.model.eval()
        vali_start_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            attention =  self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[1]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        vail_time = time.time() - vali_start_time
        self.model.train()
        return total_loss, vail_time

    def train(self, setting):
        if self.args.model == 'PromptCast':
            print("PromptCast 是 API 模型，不需要训练，自动跳过训练过程。")
            return
        train_start_time_total = time.time()
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        total_train_time = 0
        total_vali_time = 0
        total_test_time = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            attention_sender_list = []
            attention_rece_list = []

            self.model.train()
            epoch_start_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            attention =  self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[1]
                            attention_sender_list.extend(attention[0].cpu().detach().numpy())
                            attention_rece_list.extend(attention[1].cpu().detach().numpy())

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        attention =  self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[1]
                        attention_sender_list.extend(attention[0].cpu().detach().numpy())
                        attention_rece_list.extend(attention[1].cpu().detach().numpy())

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    if torch.cuda.is_available():
                        # 已分配显存 (MB)
                        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                        # 峰值显存 (MB)
                        max_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                        mem_info = f" | Mem: {allocated:.0f}MB(Max:{max_mem:.0f}MB)"
                    else:
                        mem_info = ""
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}{3}".format(
                        i + 1, epoch + 1, loss.item(), mem_info))
                    
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()


                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)
            epoch_time = time.time() - epoch_start_time
            total_train_time += epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_time = self.vali(vali_data, vali_loader, criterion)
            total_vali_time += vali_time
            test_loss, test_time = self.vali(test_data, test_loader, criterion)
            total_test_time += test_time


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            ## 到早停的最后一步进行Attention可视化
            if early_stopping.early_stop:
                print("Early stopping")
                # visualize_self_attention(
                # attention_array = np.array(attention_sender_list),
                # save_dir="/home/home_new/syc/code/Timeserise_Library/attention_visualizations",
                # file_prefix= "LCCH_attention_200_sender_imf0"
                # )
                # ## 调整时间步
                # visualize_self_attention(
                # attention_array = np.array(attention_rece_list),
                # save_dir="/home/home_new/syc/code/Timeserise_Library/attention_visualizations",
                # file_prefix= "LCCH_attention_200_rece_imf0"
                # )
                # break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # with open("result_long_term_forecast_LCCH_time.txt", 'a') as f:
        #     f.write(setting + "  \n")
        #     f.write(f'Total Train Time: {total_train_time:.2f} seconds\n')
        #     f.write(f'Total Validation Time: {total_vali_time:.2f} seconds\n')
        #     f.write(f'Total Test Time: {total_test_time:.2f} seconds\n')
        #     f.write('\n')
        # print("Trainging over!!")
        # visualize_self_attention(
        #         attention_array = np.array(attention_sender_list),
        #         save_dir="/home/home_new/syc/code/Timeserise_Library/attention_visualizations",
        #         file_prefix= "LCCH_attention_200_sender_imf0"
        #         )
        # visualize_self_attention(
        #         attention_array = np.array(attention_rece_list),
        #         save_dir="/home/home_new/syc/code/Timeserise_Library/attention_visualizations",
        #         file_prefix= "LCCH_attention_200_rece_imf0"
        #         )
        train_end_time_total = time.time()
        total_train_duration = train_end_time_total - train_start_time_total
        
        print(">>>>>>> Training Total Time: {:.2f} seconds <<<<<<<".format(total_train_duration))

        return self.model

    def test(self, setting, test=0):
        if self.args.model == 'PromptCast':
            print("PromptCast 开始执行推理测试...")
        test_data, test_loader = self._get_data(flag='test')
        if test and self.args.model != 'PromptCast':
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        test_start_time_total = time.time()
        preds = []
        trues = []
        attention_sender_list = []
        attention_rece_list = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            attention =  self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[1]
                            attention_sender_list.extend(attention[0].cpu().detach().numpy())
                            attention_rece_list.extend(attention[1].cpu().detach().numpy())

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        attention =  self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[1]
                        attention_sender_list.extend(attention[0].cpu().detach().numpy())
                        attention_rece_list.extend(attention[1].cpu().detach().numpy())


                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        #folder_path = './results/KDD/' + setting + '/'
        # folder_path = './results/LCCH/' + setting + '/'
        folder_path = './results/HGI/' + setting + '/' 
 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mae:{:.4f}, mse:{:.4f}'.format(mae, mse))
        f = open("./results/HGI/result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{:.4f}, mse:{:.4f}'.format(mae, mse))
        f.write('\n')
        f.write('\n')
        f.close()

        # visualize_self_attention(
        #         attention_array = np.array(attention_sender_list),
        #         save_dir="/home/home_new/syc/code/Timeserise_Library/attention_visualizations",
        #         file_prefix= "attention_sender"
        #         )
        # visualize_self_attention(
        #         attention_array = np.array(attention_rece_list),
        #         save_dir="/home/home_new/syc/code/Timeserise_Library/attention_visualizations",
        #         file_prefix= "attention_rece"
        #         )

        ## 批量实验整理成result.csv
        if "exp" in setting: ## 真正进行实验的时候才需要切割保存成实验结果
            model_name = re.split(r'.*exp.*?(S_|M_)',setting.split("_DYG_Oneshot",1)[0],1)[2]
            target_name = setting.split("_Third_",1)[1].split("_exp_",1)[0]
            result_csv_path = "result_long_term_forecast.csv"
            if not os.path.exists(result_csv_path):
                with open (result_csv_path,mode="w",newline="") as f:
                    writer =csv.writer(f,delimiter='\t')
                    writer.writerow(["Target", "Model", "MAE", "MSE"])
            with open (result_csv_path,mode="a",newline="") as f:
                writer = csv.writer(f,delimiter='\t')
                writer.writerow([target_name, model_name, f'{mae:.4f}', f'{mse:.4f}'])

        ##  保存两种格式
        test_end_time_total = time.time()
        total_test_duration = test_end_time_total - test_start_time_total
        
        print(">>>>>>> Test Total Time: {:.2f} seconds <<<<<<<".format(total_test_duration))
        metric_array = np.array([mae, mse, rmse, mape, mspe])
        print(preds.shape)
        print(trues.shape)
        num_features = int(self.args.c_out)
        preds_2d = preds.reshape(-1,num_features)
        trues_2d = trues.reshape(-1,num_features)
        np.savetxt(folder_path + 'metrics.csv', metric_array.reshape(1, metric_array.shape[0]), delimiter=',',\
                    fmt="%.4f",header='MAE,MSE,RMSE,MAPE,MSPE', comments='')
        ## 单变量预测
        if num_features == 1:
            np.savetxt(folder_path + 'pred.csv', preds_2d, delimiter=',',fmt="%.4f",header='Pred',comments='')
            np.savetxt(folder_path + 'true.csv', trues_2d, delimiter=',',fmt="%.4f",header='True',comments='')
        ## 多变量预测
        if num_features !=1:
            preds_header = ",".join(f'Pred_target{i}' for i in range(1,num_features+1))
            trues_header = ",".join(f'True_target{i}' for i in range(1,num_features+1))
            np.savetxt(folder_path + 'pred.csv', preds_2d, delimiter=',',fmt="%.4f",header=preds_header,comments='')
            np.savetxt(folder_path + 'true.csv', trues_2d, delimiter=',',fmt="%.4f",header=trues_header,comments='')


        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
