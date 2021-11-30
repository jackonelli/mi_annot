python src/dino/ssl_fine_tune.py \
  --data_path=${IMAGENET_PATH} \
  --arch=vit_small \
  --patch_size=16 \
  --n_last_blocks=4 \
  --pretrained_weights=/home/jakob/doktor/projects/mi_annot/code/stored_models/dino/vit_small/dino_deitsmall17_pretrain.pth \
  --evaluate \
