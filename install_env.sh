conda create -n reno python==3.11 -y
conda activate reno
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install pytorch-lightning==2.2 datasets==2.18 transformers==4.38.2 diffusers==0.30.0 hpsv2==1.2 image-reward==1.5 open-clip-torch==2.24 blobfile openai-clip setuptools==60.2 optimum scipy