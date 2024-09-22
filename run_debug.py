import subprocess

def run_model(model, data, features, seq_len, label_len, pred_len, e_layers, d_layers, \
              factor, enc_in, dec_in, c_out, d_model, d_ff, top_k, des, batch_size, itr, \
                devices, target,direction):
    command = [
        'python', '-u', 'run.py',
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--root_path', './data/DYG/',
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
        '--direction',direction
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


run_model(
    'MecHalfRouterFormer', 'DYG_data_3_sub1-2.csv', features='M', seq_len=96, label_len=48, pred_len=96, e_layers=2, 
    d_layers=1, factor=3, enc_in=5, dec_in=5, c_out=5, d_model=256, d_ff=512, top_k=5, \
          des='test', batch_size=1024, itr=1, devices='0,1,2,3', target='order1',direction='1,1,0,0,0'
          )
