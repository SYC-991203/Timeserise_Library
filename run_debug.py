import subprocess

def run_model(
             data_path, model, data, features, seq_len, label_len, pred_len, e_layers, d_layers, 
             factor, enc_in, enc_out, dec_in, c_out, d_model, d_ff, top_k, des, batch_size, itr, 
             devices, target, direction, output_attention,channel_independen):
    command = [
        'python', '-u', 'run.py',
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--root_path', './data/public',
        '--data_path', data_path,
        '--model_id', 'RopeTest',
        '--model', model,
        '--data', data,
        '--features', features,
        '--seq_len', str(seq_len),
        '--label_len', str(label_len),
        '--pred_len', str(pred_len),
        '--e_layers', str(e_layers),
        '--d_layers', str(d_layers),
        '--factor', str(factor),
        '--enc_in', str(enc_in),
        '--enc_out', str(enc_out),
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
        '--direction', direction,
        '--channel_independen','1'
       
    ]
    
    # 检查是否需要传递 --output_attention
    if output_attention:
        command.append('--output_attention')
    
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
    data_path='ETTh1.csv', model='iTransformer', data="ETTh1",
    features='S', seq_len=96, label_len=96, 
    pred_len=96, e_layers=2, 
    d_layers=1, factor=3, enc_in=7, enc_out=7, dec_in=7, c_out=1, d_model=256, d_ff=512, top_k=5, 
    des='Sens_Exp', batch_size=64, itr=1, devices='0,1,2,3,4', target='OT', direction='0,0,1,1,1',
    output_attention=False, channel_independen=1  # 注意此处为布尔值,但是是通过action开关实现的
)