#export CUDA_VISIBLE_DEVICES=0,1
# =================================================================================
# Train CTCNet
# =================================================================================

python train.py --gpus 2 --name ab16_1 --model ctcnet \
    --lr 0.0001 --beta1 0.9 --scale_factor 4 --load_size 128 \
    --dataroot "/home/ubuntu/thumb_data/"  --dataset_name celeba --batch_size 12 --total_epochs 100 \
    --visual_freq 5000 --print_freq 2000 --save_latest_freq 60000
