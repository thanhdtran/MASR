# MASR
This repo contains source code for our paper: "Adversarial Mahalanobis Distance-based Attentive Song Recommender for Automatic Playlist Continuation" published in SIGIR 2019. This source code is coming soon (expect to be pushed before July 21)!

## Data Format:
- *.rating data file: [user_id]:[playlist_id] \t track_id \t [random-position-number] \t [1]
- *.negative data file: ([user_id]:[playlist_id],[track_id]) \t [negative_track_id1] \t [negative_track_id2] ...
## Demo example:
### Training MDR and AMDR:
#### Training with MDR:

```
python -u main.py --cuda 1 --dropout 0.2 --dataset demo --epochs 50 --load_best_chkpoint 0 --model mdr --num_factors 64 --reg_mdr 0.0 --adv 0 --act_func_mdr none --data_type upt
```

#### Running with AMDR:
After training MDR, we will have best checkpoint saved at **chk_points**. The model will then automatically load the best chekpoint w.r.t the validation dataset, and use it as an initial start for adversarial learning. Without the initial learning of MDR, if you learn with adversarial learning from the sractch, we can get lower results.

```
python main.py --dataset demo --data_type upt --model mdr --num_factors  64 --reg_mdr 0.0 --load_best_chkpoint 1 --cuda 1 --epochs 50 --adv 1 --reg_noise 1.0 --eval 0 --lr 1e-3 
```

If you dont have GPU, then set ```--cuda 0```.

Similarly, we can do for MASS. After that, we can learn MASR, as a combined model, accordingly.
