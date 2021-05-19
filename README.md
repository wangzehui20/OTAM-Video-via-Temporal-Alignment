# OTAM-Video via Temporal Alignment

This code is implemented based on [Soft DTW for PyTorch in CUDA](https://github.com/Maghoumi/pytorch-softdtw-cuda) depending on [PyTorch](https://pytorch.org/) and [Numba](http://numba.pydata.org/).

## Example Usage

You can also run the included profiler/test (tested with Python v3.6), and see the speedups you'd get:

```
git clone https://github.com/Maghoumi/pytorch-softdtw-cuda
cd pytorch-softdtw-cuda
python soft_dtw_cuda.pypython soft_dtw_cuda.py
```

A sample code is already provided in the script. Here's a quick example:

```
from soft_dtw_cuda import SoftDTW

# Create the sequences
batch_size, len_x, len_y, dims = 8, 15, 12, 5
x = torch.rand((batch_size, len_x, dims), requires_grad=True)
y = torch.rand((batch_size, len_y, dims))

# Create the "criterion" object
sdtw = SoftDTW(use_cuda=True, gamma=0.1)

# Compute the loss value
loss = sdtw(x, y)  # Just like any torch.nn.xyzLoss()

# Aggregate and call backward()
loss.mean().backward()
```

## Learn More

you can learn about algorithm in  [Few-Shot Video Classification via Temporal Alignment](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf).

If you want to see alignment path between two features, you can run ...path.py in `save path` file.
