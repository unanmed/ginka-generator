# MaskGIT
python3 -u -m ginka.train_maskGIT --epochs 150 --checkpoint 10 >> output_maskGIT.log
python3 -u -m ginka.train_maskGIT --resume true --epochs 150 --checkpoint 10 --state_ginka "result/transformer/ginka-100.pth" >> output_maskGIT.log
