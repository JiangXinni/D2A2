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
model = torch.nn.DataParallel(model)
print("model done")


# dataloader
if args.dataset == 'nyu':
    train_dataset = NYU_v2_dataset(root_dir=args.dataset_dir, scale=args.scale, 
                            augment=args.augment, input_size = args.input_size)
    test_dataset = NYU_v2_dataset(root_dir=args.dataset_dir, scale=args.scale, 
                            train=False, augment=False, input_size=None)
else:
    raise NotImplementedError(f'Dataset {args.dataset} not found')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
print("dataloader done")


# trainer
if args.pretrain_path != None:
    model.load_state_dict(torch.load(args.pretrain_path))
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5,last_epoch= args.last_epoch)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5,last_epoch= args.last_epoch)

criterion = torch.nn.L1Loss()
trainer = Trainer(args, model, optimizer, scheduler, criterion, train_loader, test_loader)
print("trainer done")


### main
trainer.train()

