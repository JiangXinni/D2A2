CUDA_VISIBLE_DEVICES=0 \
python test.py \
               --model_name D2A2\
               --model_file D2A2\
               --scale 4\
               --net_path 'pretrained/D2A2_x4' \
               --dataset  'nyu'\
               --dataset_dir  '/home/lab426/Codes/jxn/Datasets/nyu_labeled'\
               # --save
