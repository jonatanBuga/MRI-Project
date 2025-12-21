# TransUNet (vendored) — Beckschen/TransUNet

This folder is intentionally **empty of upstream code** in this repository.

It is meant to contain a **local copy (or git submodule)** of the official TransUNet repository:

- https://github.com/Beckschen/TransUNet

We only need the **minimal Python modules required to import and run the segmentation model** for inference (and later training). The wrapper in our repo expects to be able to import TransUNet’s `networks/*` modules.

## Expected folder layout (important)

One of the following must exist:

### Layout 1 (recommended): upstream repo root is vendored here
```
third_party/transunet/
  networks/
  datasets/
  ...
```

### Layout 2: upstream repo is nested under `TransUNet/`
```
third_party/transunet/TransUNet/
  networks/
  datasets/
  ...
```

The wrapper (`src/models/transunet_wrapper.py`) will try both layouts automatically.

## Option A — add as a git submodule (recommended)

From the **repo root**:
```bash
mkdir -p third_party/transunet
git submodule add https://github.com/Beckschen/TransUNet third_party/transunet/TransUNet
git submodule update --init --recursive
```

This yields Layout 2.

## Option B — download ZIP and copy

1) Download ZIP from GitHub:
   - https://github.com/Beckschen/TransUNet

2) Unzip it and copy into **one of the supported layouts**:

**Layout 2 example**:
```bash
mkdir -p third_party/transunet
cp -R /path/to/TransUNet-main third_party/transunet/TransUNet
```

(Or copy the contents directly into `third_party/transunet/` for Layout 1.)

## Python dependencies (upstream)

TransUNet commonly requires extra packages beyond PyTorch. For **importing and running the model**, you will likely need at least:

```bash
pip install ml-collections einops
```

If you run into import errors from upstream code, install whatever it requires (some setups also use `opencv-python`, `scipy`, etc.). We do **not** vendor dependencies here.