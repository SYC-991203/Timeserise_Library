# models/PromptCast.py (Batch Optimized Version)

import torch
import torch.nn as nn
from openai import OpenAI
import json
import numpy as np 

class Model(nn.Module):
    """
    PromptCast - Batch Optimized Version
    使用在线 LLM 一次处理整个 batch
    """

    def __init__(self, configs):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.pred_len = configs.pred_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = configs.data_path

        # ================================
        # 你的 API 信息（任选其一）
        # ================================
        self.client = OpenAI(
            api_key="sk-a8xxxxxxxxx",
            base_url="https://api.deepseek.com/v1"               # DeepSeek
            # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # Qwen
            # base_url="https://api.openai.com/v1"                # OpenAI
        )
        self.llm_model = "deepseek-chat"   # 可换：qwen-turbo / gpt-4o-mini

        print(f"[PromptCast Batch] Using online model: {self.llm_model}")

    # ============================
    # 构建 batch prompt
    # ============================
    def get_variable_names(self):
        """
        根据datapath自动获取变量名列表
        """
        # 从datapath中提取子数据集信息
        if "sub1" in self.data_path:
            return ["Bacterial Density", "Viscosity", "Alcohol Addition", "Reducing Sugar", "Non Reducing Sugar"]
        elif "sub2" in self.data_path:
            return ["Viscosity", "Chemical Efficiency", "DO Value", "pH Online", "Air Flow"]
        elif "sub3" in self.data_path:
            return ["Bacterial Density", "Viscosity", "Chemical Efficiency", "pH Online", "Total Sugar"]
        else:
            # 默认变量名（如果无法识别）
            return [f"variable_{i}" for i in range(5)]
    def build_batch_prompt(self, batch_data, batch_size=4, chunk_idx=0, total_chunks=1):
        """
        基于PromptCast论文的方法构建prompt，支持分块预测
        """
        batch_dict = {}
        B, L, C = batch_data.shape
        
        # 限制batch大小，避免prompt过长
        actual_B = min(B, batch_size)
        
        # 自动获取变量名称
        variable_names = self.get_variable_names()
        
        # 计算当前chunk的预测范围
        chunk_size = self.pred_len // total_chunks
        start_step = chunk_idx * chunk_size
        end_step = min((chunk_idx + 1) * chunk_size, self.pred_len)
        steps_in_chunk = end_step - start_step
        
        for i in range(actual_B):
            numpy_array = batch_data[i].cpu().numpy()
            
            # 为每个序列构建自然语言描述
            seq_description = []
            
            # 为每个变量构建描述
            for var_idx in range(C):
                var_values = [round(float(x), 2) for x in numpy_array[:, var_idx]]
                var_name = variable_names[var_idx] if var_idx < len(variable_names) else f"variable_{var_idx}"
                
                # 构建类似论文中的自然语言描述
                var_desc = f"The historical {var_name} values for the past {L} time steps are: {', '.join(map(str, var_values))}."
                seq_description.append(var_desc)
            
            # 组合所有变量的描述
            full_description = " ".join(seq_description)
            batch_dict[f"seq_{i}"] = full_description

        # 构建基于论文格式的prompt
        prompt = (
            "You are an expert in industrial process time-series forecasting.\n"
            "Below are industrial process sequences with their historical observations:\n\n"
        )
        
        # 添加序列描述
        for seq_id, description in batch_dict.items():
            prompt += f"{seq_id}: {description}\n\n"
        
        # 添加预测任务说明（分块版本）
        if total_chunks > 1:
            prompt += (
                f"Based on the historical data of {L} time steps for {C} process variables, "
                f"predict ONLY the next {steps_in_chunk} time steps "
                f"(steps {start_step + 1} to {end_step}) for each sequence.\n"
                f"This is chunk {chunk_idx + 1} of {total_chunks}.\n"
            )
        else:
            prompt += (
                f"Based on the historical data of {L} time steps for {C} process variables, "
                f"predict the next {self.pred_len} time steps for each sequence.\n"
            )
        
        prompt += (
            "Return ONLY a JSON object with the same sequence keys, where each value is a list of predicted values "
            f"with shape [{steps_in_chunk}, {C}] for the {C} variables in this order: {', '.join(variable_names)}.\n\n"
            "Example output format:\n"
            "{\n"
            f'  "seq_0": [[{variable_names[0]}_1, {variable_names[1]}_1, ..., {variable_names[-1]}_1], [{variable_names[0]}_2, {variable_names[1]}_2, ...], ...],\n'
            f'  "seq_1": [[{variable_names[0]}_1, {variable_names[1]}_1, ..., {variable_names[-1]}_1], [{variable_names[0]}_2, {variable_names[1]}_2, ...], ...]\n'
            "}\n"
            "Ensure all predicted values are numerical and maintain the same units as historical data.\n"
        )
        
        return prompt

    # ============================
    # 一次调用处理整个 batch
    # ============================
    def call_llm_batch(self, prompt):
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        print("GET RESPONSE")
        return resp.choices[0].message.content

    # ============================
    # forward
    # ============================
    def parse_llm_response(self, result_text, expected_steps, expected_channels):
        """解析LLM响应"""
        try:
            json_start = result_text.index("{")
            json_end = result_text.rindex("}") + 1
            result_json = json.loads(result_text[json_start:json_end])
            return result_json
        except:
            print("Failed to parse LLM response, returning zeros")
            return None
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: (B, L, C)
        输出: (B, pred_len, C)
        使用分块预测防止输出过长
        """
        B, _, C = x_enc.shape
        
        # 初始化输出张量
        output = torch.zeros(B, self.pred_len, C).to(self.device)
        
        # 计算分块参数
        chunk_size = getattr(self, 'chunk_size', 24)  # 默认每个chunk预测24步
        total_chunks = (self.pred_len + chunk_size - 1) // chunk_size
        
        print(f"Using chunked prediction: {total_chunks} chunks, {chunk_size} steps per chunk")
        
        # 为每个chunk进行预测
        for chunk_idx in range(total_chunks):
            start_step = chunk_idx * chunk_size
            end_step = min((chunk_idx + 1) * chunk_size, self.pred_len)
            steps_in_chunk = end_step - start_step
            
            print(f"Predicting chunk {chunk_idx + 1}/{total_chunks} (steps {start_step + 1}-{end_step})")
            
            # 构建当前chunk的prompt
            actual_batch_size = min(B, 4)
            prompt = self.build_batch_prompt(
                x_enc, 
                batch_size=actual_batch_size,  # 限制batch大小
                chunk_idx=chunk_idx, 
                total_chunks=total_chunks
            )
            
            # 调用LLM
            result_text = self.call_llm_batch(prompt)
            
            # 解析响应
            result_json = self.parse_llm_response(result_text, steps_in_chunk, C)
            
            if result_json is None:
                # 如果解析失败，使用简单fallback：线性外推
                self._apply_fallback_prediction(output, x_enc, start_step, end_step, chunk_idx)
                continue
            
            # 填充预测结果
            self._fill_predictions(output, result_json, x_enc, start_step, end_step, chunk_idx,actual_batch_size)
        
        return output

    def _apply_fallback_prediction(self, output, x_enc, start_step, end_step, chunk_idx):
        """应用fallback预测策略 - 线性外推"""
        B, L, C = x_enc.shape
        steps_in_chunk = end_step - start_step
        
        # 使用最后几个时间点的线性外推
        if L >= 2:
            # 计算最后两个时间点的变化趋势
            last_value = x_enc[:, -1:, :]  # (B, 1, C)
            second_last_value = x_enc[:, -2:-1, :]  # (B, 1, C)
            trend = last_value - second_last_value  # (B, 1, C)
            
            # 线性外推
            for step in range(steps_in_chunk):
                output[:, start_step + step] = (last_value + trend * (step + 1)).squeeze(1)
        else:
            # 如果历史数据不足，直接使用最后一个值
            last_values = x_enc[:, -1:, :].repeat(1, steps_in_chunk, 1)
            output[:, start_step:end_step] = last_values.to(self.device)
        
        print(f"Applied fallback prediction for chunk {chunk_idx + 1}")

    def _fill_predictions(self, output, result_json, x_enc, start_step, end_step, chunk_idx, actual_batch_size):
        """将解析的预测结果填充到输出张量中"""
        B = output.shape[0]  # 总batch大小
        steps_in_chunk = end_step - start_step
        
        success_count = 0
        # 只处理实际被发送给LLM的序列
        for i in range(actual_batch_size):
            key = f"seq_{i}"
            if key in result_json:
                chunk_pred = result_json[key]
                
                if isinstance(chunk_pred, list) and len(chunk_pred) >= steps_in_chunk:
                    chunk_tensor = torch.tensor(chunk_pred[:steps_in_chunk], dtype=torch.float32)
                    
                    if chunk_tensor.dim() == 2 and chunk_tensor.shape[1] == output.shape[2]:
                        output[i, start_step:end_step] = chunk_tensor.to(output.device)
                        success_count += 1
                    else:
                        print(f"Shape mismatch for {key}: expected ({steps_in_chunk}, {output.shape[2]}), got {chunk_tensor.shape}")
                        # 对于形状不匹配的，使用fallback
                        self._apply_single_fallback(output, x_enc, i, start_step, end_step)
                else:
                    print(f"Length mismatch for {key}: expected {steps_in_chunk}, got {len(chunk_pred)}")
                    # 对于长度不匹配的，使用fallback
                    self._apply_single_fallback(output, x_enc, i, start_step, end_step)
            else:
                print(f"Missing key {key} in chunk {chunk_idx + 1}")
                # 对于缺失的key，使用fallback
                self._apply_single_fallback(output, x_enc, i, start_step, end_step)
        
        # 对于没有被发送给LLM的序列，使用fallback
        for i in range(actual_batch_size, B):
            self._apply_single_fallback(output, x_enc, i, start_step, end_step)
        
        print(f"Chunk {chunk_idx + 1}: {success_count}/{actual_batch_size} sequences successfully predicted")

    def _apply_single_fallback(self, output, x_enc, seq_idx, start_step, end_step):
        """对单个序列应用fallback预测"""
        L = x_enc.shape[1]
        steps_in_chunk = end_step - start_step
        
        if L >= 2:
            # 线性外推
            last_value = x_enc[seq_idx, -1:, :]  # (1, C)
            second_last_value = x_enc[seq_idx, -2:-1, :]  # (1, C)
            trend = last_value - second_last_value  # (1, C)
            
            for step in range(steps_in_chunk):
                output[seq_idx, start_step + step] = (last_value + trend * (step + 1)).squeeze(0)
        else:
            # 直接使用最后一个值
            last_value = x_enc[seq_idx, -1, :]  # (C,)
            output[seq_idx, start_step:end_step] = last_value.unsqueeze(0).repeat(steps_in_chunk, 1)
