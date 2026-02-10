#!/usr/bin/env bash
set -euo pipefail

# ====== Settings ======
ROOT_DIR="/projects/betw/msalunkhe/data/imagenet"
URL_TRAIN="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
URL_VAL="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"
URL_TEST="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar"

# ====== Create directories ======
mkdir -p "$ROOT_DIR"/{tars,train,val,test}

cd "$ROOT_DIR"

# ====== Download tar files ======
echo "==> Downloading tar files into $ROOT_DIR/tars"
wget -c "$URL_TRAIN" -O "tars/ILSVRC2012_img_train.tar"
wget -c "$URL_VAL"   -O "tars/ILSVRC2012_img_val.tar"
wget -c "$URL_TEST"  -O "tars/ILSVRC2012_img_test_v10102019.tar"

# ====== Extract top-level archives ======
echo "==> Extracting train tar (creates many per-class .tar files)..."
tar -xf "tars/ILSVRC2012_img_train.tar" -C train

echo "==> Extracting val tar..."
tar -xf "tars/ILSVRC2012_img_val.tar" -C val

echo "==> Extracting test tar..."
tar -xf "tars/ILSVRC2012_img_test_v10102019.tar" -C test

# ====== Extract nested train class tars ======
echo "==> Extracting nested per-class train tars..."
cd train
shopt -s nullglob
for f in *.tar; do
  d="${f%.tar}"
  mkdir -p "$d"
  tar -xf "$f" -C "$d"
  rm -f "$f"
done
cd ..

echo "==> Done."
echo "Dataset structure:"
echo "  $ROOT_DIR/train/<wnid>/*.JPEG"
echo "  $ROOT_DIR/val/*.JPEG   (note: val is not sorted into wnid folders by default)"
echo "  $ROOT_DIR/test/*.JPEG"
