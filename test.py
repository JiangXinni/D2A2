from importlib import import_module

from utils import *
from datasets import *
from option import args

setup_seed(20)



# model
module = import_module('models.' + args.model_file)
if args.model_name == 'D2A2':
    model = module.D2A2(args).cuda()
else:
    raise NotImplementedError(f'Model {args.model_name} in file {args.model_file} not found')
model = nn.DataParallel(model)
model.load_state_dict(torch.load(args.net_path))
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

