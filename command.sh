python src/dino/gen_dino_imagenet_features.py \
  --data_path=${DINO_FEATURES_PATH} \
# python src/dino/gen_dino_imagenet_features.py \
#   --data_path=${IMAGENET_PATH} \
#   --arch=vit_small \
#   --patch_size=16 \
#   --n_last_blocks=4 \
#   --batch_size_per_gpu=512 \
#   --pretrained_weights=/home/jakob/doktor/projects/mi_annot/code/stored_models/dino/vit_small/dino_deitsmall16_pretrain.pth \

# python src/dino/eval_linear.py \
#   --data_path=${IMAGENET_PATH} \
#   --arch=vit_small \
#   --patch_size=16 \
#   --n_last_blocks=4 \
#   --batch_size_per_gpu=256 \
#   --pretrained_weights=/home/jakob/doktor/projects/mi_annot/code/stored_models/dino/vit_small/dino_deitsmall16_pretrain.pth \
