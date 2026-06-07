import subprocess
import json
def run_model(model, data, features, seq_len, label_len, pred_len, e_layers, d_layers, \
              factor, enc_in, dec_in, c_out, d_model, d_ff, top_k, des, batch_size, itr, \
              devices, target, direction,moe_logits_init_dict,loss,temp): # <--- 添加字典参数
    
    # 步骤 1: 将传入的 Python 字典转换为 JSON 字符串,私有数据集地址在'--root_path', './data/DYG/KDD',公开数据集在./data/pulic
    moe_logits_json = json.dumps(moe_logits_init_dict)
    command = [
        'python', '-u', 'run.py',
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--root_path', './data/DYG/KDD',
        '--data_path', data,
        '--model_id', 'DataArg',
        '--model', model,
        '--data', 'DYG_base',
        '--features', features,
        '--seq_len', str(seq_len),
        '--label_len', str(label_len),
        '--pred_len', str(pred_len),
        '--e_layers', str(e_layers),
        '--d_layers', str(d_layers),
        '--factor', str(factor),
        '--enc_in', str(enc_in),
        '--dec_in', str(dec_in),
        '--c_out', str(c_out),
        '--d_model', str(d_model),
        '--d_ff', str(d_ff),
        '--top_k', str(top_k),
        '--des', des,
        '--batch_size', str(batch_size),
        '--itr', str(itr),
        '--devices', devices,
        '--target', target,
        '--direction',direction,
        '--moe_logits_init', moe_logits_json,
        '--loss',loss,
        '--temp',temp,
        
    ]
    
    # 运行命令行
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 检查结果
    if result.returncode == 0:
        print("命令执行成功，输出：")
        print(result.stdout)
    else:
        print("命令执行失败，错误：")
        print(result.stderr)

moe_init_params = {
    "0": {0: 5.0, 1: 1.0, 2: -5.0},  
    "1": {0: 1.0, 1: 5.0, 2: -5.0},  
    "2": {0: -5.0, 1: 1.0, 2: 5.0},  
    "3": {0: -5.0, 1: 4.0, 2: 5.0},  
    "4": {0: -5.0, 1: 4.0, 2: 5.0}
}
run_model(
    'HGI_PerHKGNN', 'DYG_data_3_sub1.csv', features='M', seq_len=96, label_len=48, pred_len=96, e_layers=2, 
    d_layers=1, factor=3, enc_in=5, dec_in=5, c_out=5, d_model=512, d_ff=512, top_k=5, \
          des='test', batch_size=256, itr=1, devices='1,2,3,6', target='zs',direction='1,1,2,0,0',
          moe_logits_init_dict=moe_init_params,loss='MAE',temp='2.0'
          )
## sub1
# moe_init_params = {
#     "0": [5.0, -5.0],  # [独立权重，共享权重]
#     "1": [5.0, -5.0],  # Group 1: 强烈共享性 (E1)
#     "2": [0.0, 0.0]    # Group 2: 中立
# }
# direction='1,1,2,0,0'
# sub2
# moe_init_params = {
#     "0": [0.0, 0.0],  # [独立权重，共享权重]
#     "1": [-5.0, 5.0],  # Group 1: 强烈共享性 (E1)
#     "2": [-5.0, 5.0]    # Group 2: 中立
# }
# direction='1,1,2,0,2'
# sub3:Bacterial Density,Viscosity,Chemical Efficiency,pH Online,Total Sugar
# moe_init_params = {
#     "0": [-3.0, 3.0],  # [独立权重，共享权重]
#     "1": [0.0,0.0],  # Group 1: 强烈共享性 (E1)
# }
# direction='0,0,0,1,1'