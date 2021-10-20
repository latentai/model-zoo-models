# USAGE
"""
python preprocess_imagenet.py --img_dir <Path to imgs folder> --out_dir <where to save prerprocessed images>
"""
import argparse
import os

import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--img_dir",
    dest='imgdir',
    type=str
)
parser.add_argument(
    "--out_dir",
    dest='outdir',
    type=str,
    default='./imagenet_new'
)
args = parser.parse_args()

imgdir = args.imgdir
outdir = args.outdir
assert os.path.exists(imgdir), \
    AssertionError("Img dir not found")
if not os.path.exists(outdir):
    os.mkdir(outdir)
if not imgdir.endswith("/"):
    imgdir += '/'
if not outdir.endswith("/"):
    outdir += '/'
fns_all = os.listdir(imgdir)
fns_all.sort()
fns_all = [imgdir + fn for fn in fns_all]
tot_imgs = len(fns_all)
bsize = 1000
for i in range(tot_imgs):
    if i % bsize == 0:
        print("%d/%d" % (i, tot_imgs))

    img = cv2.imread(fns_all[i])
    # RESIZE
    height, width, _ = img.shape
    new_H = height * 256 // min(img.shape[:2])
    new_W = width * 256 // min(img.shape[:2])
    img = cv2.resize(img, (new_W, new_H),
                     interpolation=cv2.INTER_CUBIC)

    # CROP
    height, width, _ = img.shape
    startx = width // 2 - (224 // 2)
    starty = height // 2 - (224 // 2)
    img = img[starty:starty + 224, startx:startx + 224]
    assert img.shape[0] == 224 and img.shape[1] == 224, \
        AssertionError("%d %d %d" % (img.shape, height, width))

    cv2.imwrite(fns_all[i].replace(imgdir, outdir), img)
