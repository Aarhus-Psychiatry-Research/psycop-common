# The only ground truth on how to set things up on Overtaci

pip install -r requirements.txt -r dev-requirements.txt -r gpu-requirements.txt
conda install pytorch=2.1.0 pytorch-cuda=12.1 -c https://exrhel0371.it.rm.dk/api/repo/pytorch -c https://exrhel0371.it.rm.dk/api/repo/nvidia -c https://exrhel0371.it.rm.dk/api/repo/anaconda --override-channels --insecure -y

# test environment
python -c "import torch; t=torch.tensor(1); t.to(torch.device('cuda'))"