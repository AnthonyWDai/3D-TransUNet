conda create --name transunet python=3.11 -y
conda activate transunet
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install numpy==1.23
pip install monai
pip install matplotlib batchgenerators pandas SimpleITK medpy tqdm

pip install segmentation_models_pytorch monai einops SimpleITK # installed
pip install pyyaml einops adamp gco-wrapper medpy nibabel tensorboardX tqdm ml_collections # in arnold, but now installed in venv
pip install fvcore

git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
git checkout nnunetv1
pip install -e .

cd ..
conda install -c nvidia cuda-toolkit
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation # solved no module name "torch"