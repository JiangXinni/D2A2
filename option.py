import argparse


parser = argparse.ArgumentParser()


# General setting
parser.add_argument('--model_name', type=str, default='D2A2', 
                    help='choose model class name')
parser.add_argument('--model_file', type=str, default='D2A2', 
                    help='choose model file')
parser.add_argument('--scale', type=int, default=4, 
                    help='scale factor')
parser.add_argument('--dataset', type=str, default='nyu',
                    help='dataset nyu, middlebury, lu or rgbdd')  
parser.add_argument('--dataset_dir', type=str, default='',
                    help='dataset root dir')   
parser.add_argument('--pretrain_path', type=str, default=None,
                    help='load pretrained model parameters')                  

# Model setting
parser.add_argument('--n_resblocks', type=int, default=4,
                    help='The number of residual blocks in each stage')
parser.add_argument('--n_feats', type=int, default=64,
                    help='The number of channels in network')
parser.add_argument('--res_scale', type=float, default=1.,
                    help='Residual scale')

# Train setting
parser.add_argument('--input_size', type=int, default=256, 
                    help='input size')
parser.add_argument('--epoch', type=int, default=500, 
                    help='max epoch')
parser.add_argument('--batch_size', type=int, default=4, 
                    help='batch size')
parser.add_argument('--augment', type=bool, default=True,
                    help='data augment')
parser.add_argument('--lr', type=float, default='0.0001',  
                    help='learning rate')
parser.add_argument('--step_size', type=int, default=5000000, 
                    help='learning rate step size')
parser.add_argument('--trainresult', type=str, default='./result/trainresult/', 
                    help=' train result file path')
parser.add_argument('--last_epoch', type=int, default=-1,
                    help='train epoch')

# Test setting 
parser.add_argument('--save', action='store_true',
                    help='save depth SR or not') 
parser.add_argument('--testresult', type=str, default='./result/testresult/', 
                    help=' train result file path')
parser.add_argument('--net_path', type=str, default=None,
                    help='load pretrained model parameters') 



args = parser.parse_args()
# print(args)

