# MASR
This repo contains source code for our paper: "Adversarial Mahalanobis Distance-based Attentive Song Recommender for Automatic Playlist Continuation" published in SIGIR 2019. This source code is coming soon (expect to be pushed before July 21)!

## Demo example:
### Running MDR:
#### Running with MDR:

```
python -u main.py --cuda 1 --dropout 0.2 --dataset demo --epochs 50 --load_best_chkpoint 0 --model mdr --num_factors 64 --reg_mdr 0.0 --adv 0 --act_func_mdr none --data_type upt
```

#### Running with AMDR:
After training MDR, we will have best checkpoint saved at **chk_points**. The model will then automatically load the best chekpoint w.r.t the validation dataset, and use it as an initial start for adversarial learning. Without the initial learning of MDR, if you learn with adversarial learning from the sractch, we can get lower results.

```
python -u main.py --cuda 1 --dropout 0.2 --dataset demo --epochs 50 --load_best_chkpoint 0 --model mdr --num_factors 64 --reg_mdr 0.0 --adv 0 --act_func_mdr none --data_type upt
```
