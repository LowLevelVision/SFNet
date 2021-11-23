# Spatial-Aware Fully-Cascaded Network for Motion Deblurring

## Dependencies
python
```
conda create -n sfnet python=3.7
conda activate sfnet
```
pytorch
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```
warmup_scheduler
```
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```

## Training
Please download GoPro dataset (https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view) into './dataset', and train the model by running the following command.
```
python train.py
```

## Testing
Please download pretrained model (https://drive.google.com/file/d/1KKv-6ccdsMN32JA715LoXGOCLt7twpqE/view) into './ckpts', and copy test samples into './test_samples'. Then running the following command.
```
python test.py
```

## Results
The deblurring results of SFNet on GoPro test set: https://drive.google.com/file/d/1VuoNvnFtmkOJelztP4sKRNjIb6PuKVcE/view

The results of visual comparisons: https://drive.google.com/file/d/15kUXcQog5hb7BCwsKHkHH0qeEk8hQwZ2/view
