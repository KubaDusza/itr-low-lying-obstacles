#!/usr/bin/env bash
# Put ISOD into the ESANet clone so the published RGB-D model trains on our data unchanged.
#
#   bash setup.sh [root]
#
# Idempotent. Three steps: register the dataset inside the clone, export the data into the
# layout the loader expects, and fetch the ImageNet-pretrained ResNet-34 NBt1D encoder the
# authors publish (their encoder is a non-standard ResNet variant, so torchvision's weights
# do not fit it).
set -eu
ROOT=${1:-/data/cs_courses/jdusza_itr}
ESA=$ROOT/esanet
PY=/opt/venv/jdusza_venv/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)

[ -d "$ESA" ] || git clone -q --depth 1 https://github.com/TUI-NICR/ESANet.git "$ESA"

# 1. the dataset module
mkdir -p "$ESA/src/datasets/isod"
cp "$HERE/isod_dataset.py" "$ESA/src/datasets/isod/pytorch_dataset.py"
cat > "$ESA/src/datasets/isod/__init__.py" <<'EOF'
from .pytorch_dataset import ISOD, ISODBase
EOF

$PY - "$ESA" <<'EOF'
import pathlib, sys
esa = pathlib.Path(sys.argv[1])

p = esa / "src/datasets/__init__.py"
s = p.read_text()
if "isod" not in s:
    s += "from .isod import ISOD\nfrom .isod import ISODBase\n"
    p.write_text(s)
    print("registered ISOD in src/datasets/__init__.py")

p = esa / "src/prepare_data.py"
s = p.read_text()
if "ISOD" not in s:
    s = s.replace("from src.datasets import NYUv2",
                  "from src.datasets import ISOD\nfrom src.datasets import NYUv2")
    s = s.replace("""    if args.dataset == 'sunrgbd':""",
                  """    if args.dataset == 'isod':
        Dataset = ISOD
        dataset_kwargs = {'n_classes': 2}
        valid_set = 'valid'
    elif args.dataset == 'sunrgbd':""")
    p.write_text(s)
    print("registered ISOD in src/prepare_data.py")

p = esa / "src/args.py"
s = p.read_text()
if "'isod'" not in s:
    s = s.replace("choices=['sunrgbd',", "choices=['isod',\n                                   'sunrgbd',")
    assert "'isod'" in s, "could not add isod to the --dataset choices"
    p.write_text(s)
    print("added isod to the --dataset choices")

p = esa / "train.py"
s = p.read_text()
if "ESANET_SEED" not in s:
    # ESANet has no seed argument, so repeated runs would differ for uncontrolled reasons.
    # Three lines give it one, read from the environment, leaving everything else as published.
    s = s.replace("""def train_main():
    args = parse_args()""", """def train_main():
    args = parse_args()

    seed = int(os.environ.get('ESANET_SEED', 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f'ESANET_SEED={seed}')""")
    assert "ESANET_SEED" in s, "could not add the seed to train.py"
    if "\nimport random\n" not in s:
        s = s.replace("import os\n", "import os\nimport random\n", 1)
    p.write_text(s)
    print("added ESANET_SEED to train.py")
EOF

# 2. the data
cd "$HERE/.."
ITR_DATA=$ROOT PYTHONPATH="$HERE/.." $PY esanet/export_isod.py --out "$ROOT/esanet_isod"

# 3. the published ImageNet-pretrained encoder
mkdir -p "$ESA/trained_models"
if [ ! -f "$ESA/trained_models/imagenet/r34_NBt1D.pth" ]; then
  $PY -m pip install -q gdown 2>&1 | tail -1 || true
  (cd "$ESA/trained_models" && \
   "$($PY -c 'import sys,os;print(os.path.dirname(sys.executable))')/gdown" \
     1neUb6SJ87dIY1VvrSGxurVBQlH8Pd_Bi -O imagenet_r34_NBt1D.tar.gz && \
   tar xzf imagenet_r34_NBt1D.tar.gz && rm -f imagenet_r34_NBt1D.tar.gz)
fi
ls -la "$ESA/trained_models/imagenet/" 2>/dev/null || echo "NOTE: encoder weights missing"
echo SETUP_DONE
