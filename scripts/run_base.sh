export HF_ENDPOINT=https://hf-mirror.com

python main.py \
    --model-path /home/vdig/public_data/llava-v1.6-mistral-7b \
    --dataroot /home/vdig/wangxinhao/bevperception/data/nuscenes-mini \
    --version v1.0-mini \
    --method openemma