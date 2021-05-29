#!/usr/bin/env bash
python classify.py --model model10A --epoch 350  --batch_size 128 --msg bs128_test1
python classify.py --model model10A --epoch 350  --batch_size 128 --msg bs128_test2
python classify.py --model model10A --epoch 350  --batch_size 128 --learning_rate 0.1 --msg bs128_lr0.1_test3
python classify.py --model model11F --epoch 350  --batch_size 128 --msg bs128_test1
python classify.py --model model11F --epoch 350  --batch_size 128 --msg bs128_test2
python classify.py --model model11G --epoch 350  --batch_size 64 --msg bs64_test1
python classify.py --model model11G --epoch 350  --batch_size 64 --msg bs64_test2
python classify.py --model model14A --epoch 350  --batch_size 64 --msg bs64_test1
python classify.py --model model14A --epoch 350  --batch_size 64 --msg bs64_test2

