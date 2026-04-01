export HF_ENDPOINT=https://hf-mirror.com

python main.py \
    --model-path /mnt/bn/yufei1900/wangxinhao/paper/checkpoint/llava-v1.6-mistral-7b \
    --dataroot /mnt/bn/yufei1900/wangxinhao/paper/data/nuscenes-mini \
    --version v1.0-mini \
    --output-dir ./output \
    --method openemma