
import argparse, csv
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LinearPFN')
    parser.add_argument('--data_version', type=str, default='0')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--c_min', type=int, default=32)
    parser.add_argument('--c_max', type=int, default=1536)

    args = parser.parse_args()

    csv_path = f'{args.model}/v{args.data_version}/sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_df{args.d_ff}/C{args.c_min}-{args.c_max}_Q1-32'
    metric_dict = {}

    with open(f'./results/{csv_path}/results.csv', mode='r', newline='') as file:
        csv_dict_reader = csv.DictReader(file)
        for row in csv_dict_reader:
            # print(row['column_name']) # Access data by column name
            if row['train_budget'] == '1.0':
            # print(row) # Prints the entire row as an OrderedDict
                metric_dict[f'{row["dataset"]}/mse'] = row['mse']
                metric_dict[f'{row["dataset"]}/mae'] = row['mae']
    
    hp_dict = {
        'model': args.model, 'data_version': args.data_version, 'c_min': args.c_min, 'c_max': args.c_max,
        'd_model': args.d_model, 'n_heads': args.n_heads, 'L': args.e_layers, 'd_ff': args.d_ff
    }

    log_dir = f'./training/tb_test/{args.seq_len}_{args.pred_len}'
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_hparams(hparam_dict=hp_dict, metric_dict=metric_dict)
    writer.close()