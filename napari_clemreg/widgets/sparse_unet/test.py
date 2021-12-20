#!/usr/bin/env python3
# coding: utf-8
import argparse
import imageio
from model import SparseUnet
from tqdm import tqdm
import os

def coords(s):
    try:
        x, y = map(int, s.split(','))
        return (x, y)
    except:
        raise argparse.ArgumentTypeError('Shape must be x,y! E.g. --shape 512,512')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--shape', type=coords, default=(512,512))
    parser.add_argument('--tile_shape', type=coords, default=(512,512))

    args = parser.parse_args()

    model = SparseUnet(shape=(args.shape[0], args.shape[1], 1))
    model.load(args.ckpt)

    img_lst = sorted(glob.glob(os.path.join(args.test_dir, '*.png')))

    for i in tqdm(range(len(img_lst))):
        img_pred = model.predict(imageio.imread(img_lst[i]),
                                 tile_shape=args.tile_shape)
        imageio.imwrite(os.path.join(args.out_dir, os.path.basename(img_lst[i])),
                        img_pred.astype('float'))
