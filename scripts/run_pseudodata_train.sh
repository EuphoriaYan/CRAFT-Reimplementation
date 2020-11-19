
export CUDA_VISIBLE_DEVICES=1

python trainPseudodata.py \
--batch_size 16 \
--lr 3e-4 \
--momentum 0.9 \
--weight_decay 5e-4 \
--epochs 50 \
--test_folder dataset/chinese_books \
--show_time \
