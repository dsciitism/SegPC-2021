pip install -U torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full
rm -rf mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .
pip install Pillow==7.0.0
cd ..