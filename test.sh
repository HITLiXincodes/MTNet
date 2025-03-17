python test.py --gpus 1 --model ctcnet --name final1 \
    --load_size 128 --dataset_name single --dataroot "/home/ubuntu/thumb_data/" \
    --pretrain_model_path "/home/ubuntu/hog/Ablation4-encry2/checkpoints/ab16_1/iter_200000_net_G.pth" \
    --save_as_dir "/home/ubuntu/hog/Ablation4-encry2/ff/"
