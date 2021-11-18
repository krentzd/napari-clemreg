#!/usr/bin/env python3
# coding: utf-8
import argparse
from model import SparseUnet

def coords(s):
    try:
        x, y = map(int, s.split(','))
        return (x, y)
    except:
        raise argparse.ArgumentTypeError('Shape must be x,y! E.g. --shape 512,512')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--shape', type=coords, default=(512,512))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dense_annotations', action='store_true')
    args = parser.parse_args()

    model = SparseUnet(shape=(args.shape[0], args.shape[1], 1))

    model.train(train_dir=args.train_dir,
                val_dir=args.val_dir,
                out_dir=args.out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                dense=args.dense_annotations)
