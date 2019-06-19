# Text Recognition Library based on PyTorch
======================================

This software implements the CRNN and DenseNet-CTC in pytorch.
Origin repository could be found in [crnn](https://github.com/meijieru/crnn.pytorch) [densenet-ctc](https://github.com/zhiqwang/crnn.pytorch)

## Dependence
----------
* python 2.7
* pytorch 1.0.1
* opencv >= 3.4.0
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)

## Train a new model
-----------------
```bash
./train.sh
```

## Run demo
--------
A demo program can be found in ``eval.py``. Then launch the demo by:
```python
    python eval.py
```

The demo reads images and recognizes its text content.
