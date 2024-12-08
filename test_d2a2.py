import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from importlib import import_module
from utils import *
from datasets import *
from option import args


#choice the model

if args.scale==4:
    from models.D2A2_depthanything import D2A2
elif args.scale==8:
    from models.D2A2_depthanything_L_scale8 import D2A2
elif args.scale==16:
    from models.D2A2_depthanything_L_scale16 import D2A2

setup_seed(20)
#
w=torch.load(args.net_path)
for key in w.keys():
    print(f"- {key}")
model=D2A2(args).cuda()


# print(model)
n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
if n_gpus > 1:
    model = torch.nn.DataParallel(model)


weights=torch.load(args.net_path)
weights_dict = {}
for k, v in weights.items():
    new_k = k.replace('module.', '') if 'module' in k else k
    weights_dict[new_k] = v

model.load_state_dict(weights_dict)
print("model done")

# dataloader
if args.dataset == 'nyu':
    test_dataset = NYU_v2_dataset(root_dir=args.dataset_dir,scale = args.scale,train = False,augment = False,input_size = None)
elif args.dataset == 'lu':
    test_dataset = LU_dataset(root_dir=args.dataset_dir,scale = args.scale)
elif args.dataset == 'middlebury':
    test_dataset = Middlebury_dataset(root_dir=args.dataset_dir,scale = args.scale)
elif args.dataset == 'rgbdd':
    test_dataset = RGBDD_dataset(root_dir=args.dataset_dir,scale = args.scale)
else:
    raise NotImplementedError(f'Dataset {args.dataset} not found')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
print("dataloader done")


# tester
tester = Tester(args, model, test_loader)
print("tester done")


### main
tester.validate()

