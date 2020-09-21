
export CUDA_VISIBLE_DEVICES=1

python trainPseudodata.py \
--batch_size 24 \
--lr 3.2768e-5 \
--momentum 0.9 \
--weight_decay 5e-4 \
--epochs 200 \
--test_folder dataset/chinese_books \
