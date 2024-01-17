CUDA_VISIBLE_DEVICES=0 \
python train.py \
               --model_name D2A2\
               --model_file D2A2\
               --scale 4\
               \
               --n_resblocks 4\
               --input_size 256\
               --epoch 500 \
               --batch_size 4\
               --augment 'True' \
               --dataset_dir  '/home/lab426/Codes/jxn/Datasets/nyu_labeled'\
               --dataset  'nyu'\
               --lr   '0.0001' \
               --n_feats 64