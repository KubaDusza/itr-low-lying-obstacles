#!/usr/bin/env bash
set -euo pipefail
cd /mnt/gpu_storage/cs_courses/jdusza_itr/sam_v2_20261121
trap 'printf "%s\n" "$?" > esa210_setup.exit' EXIT
export UV_CACHE_DIR="$PWD/package-cache" UV_LINK_MODE=copy
/opt/venv/jdusza_itr_prompt_v2/bin/uv pip install --python /opt/venv/jdusza_itr_prompt_v2/bin/python 'numpy==2.2.6' scipy 'pandas<3' matplotlib h5py cityscapesscripts mock
