python train_biofouling.py biofouling /media/data/seg_dataset --mode close --init_net imagenet
python infer.py /media/data/seg_dataset data/snapshots/biofouling_200_closedform.pth.tar --val_orig --norm

python train_corrosion.py corrosion /media/data/seg_dataset --mode close --init_net imagenet
python infer_corrosion.py /media/data/seg_dataset data/snapshots/corrosion_500_closedform.pth.tar --val_orig --norm

