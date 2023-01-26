# Dynamically Expandable Graph Convolution for Streaming Recommendation.

Note: We will continue to update and improve the code and related docs.

## Environment requirement

  See requirement.txt

## Log file directory

`-log-files` sets where the results logs are saved

`-load_save_path_prefix` sets top-level directory for saved models

`log_folder` sets the sub-directory for saved models

## Command examples

* **Training baselines**

> Fine-tune:
> `python -u run_baselines_segments.py -d Taobao2014 -bm MGCCF -alg Finetune -de 0 -e 25 -train_mode sep -log_folder test -log  test_Finetune -save_cp b0_100e  -patience 2 -lr 1e-3 `

> TWP:
> `python -u run_baselines_segments.py -d Taobao2014 -bm MGCCF -alg TWP -de 1 -e 25 -train_mode sep -log_folder test -log test_TWP -save_cp b0_100e -mse 100 -local_distill 1e4 -local_mode LSP_s -patience 2 -lr 1e-3`

> GraphSAIL:
> `python -u run_baselines_segments.py -d Taobao2014 -bm MGCCF -alg GraphSAIL -de 0 -e 25 -train_mode sep -log_folder test -log test_GraphSAIL -save_cp b0_100e -mse 100 -local_distill 1e4 -local_mode LSP_s -global_distill 1e4 -global_k 50,50 -global_tau 1 -patience 2 -lr 1e-3 `

> Inverse Degree Sampling:
> `python -u run_baselines_segments.py -d Taobao2014 -bm MGCCF -alg Inverse_Sampling -de 2 -e 20  -train_mode sep -log_folder test -log test_Inverse_Sampling -save_cp b0_100e  -rs full -union_mode snu -replay_ratio 0.2 -sampling_mode inverse_deg -patience 2 -lr 1e-3`

> SGCT:
> `python -u run_baselines_segments.py -d Taobao2014 -bm MGCCF -alg SGCT -de 3 -e 25  -train_mode sep -log_folder test -log test_SGCT -save_cp b0_100e -layer_wise 0 -contrastive_mode 'Single' -lambda_contrastive 1000,0,0 -con_positive 15 -con_ratios 2,1,2,0,0,0,0   -patience 2 -lr 1e-3`

> MGCT:
> `python -u run_baselines_segments.py -d Taobao2014 -bm MGCCF -alg MGCT -de 0 -e 25  -train_mode sep -log_folder test -log test_MGCT -save_cp b0_100e -layer_wise 0 -contrastive_mode 'Multi' -lambda_contrastive 100,0,0 -con_positive 15 -con_ratios 2,1,2,1,1,1,1 -patience 2 -lr 1e-3`

> LWC-KD:
> `python -u run_baselines_segments.py -d Taobao2014 -bm MGCCF -alg LWC_KD -de 0 -e 25  -train_mode sep -log_folder test -log test_LWC_KD -save_cp b0_100e -layer_wise 1 -contrastive_mode 'Multi' -lambda_contrastive 100,100,1000 -con_positive 15 -con_ratios 2,1,2,1,1,1,1 -patience 2 -lr 1e-3 `

> ContinualGNN:
> `python -u run_baselines_segments.py -d Taobao2014 -bm MGCCF -alg ContinualGNN -de 0 -e 25 -train_mode sep -log_folder test -log test_ContinualGNN -save_cp b0_100e -mse 100 -local_distill 1e4 -local_mode LSP_s -rs full -union_mode snu -replay_ratio 0.2 -sampling_mode uniform  -patience 2  -lr 1e-3 -first_segment_time 18 -last_segment_time 48`

> Uniform Experience Replay:
> `python -u run_baselines_segments.py -d Taobao2014 -bm MGCCF -alg uniform_sampling -de 0 -e 20  -train_mode sep -log_folder test -log test_uniform_sampling -save_cp b0_100e -rs full -union_mode snu -replay_ratio 0.2 -sampling_mode uniform  -patience 2  -lr 1e-3`

> Full Batch Replay:
> `python -u run_baselines_segments.py -d Taobao2014 -bm MGCCF -alg Full_batch -de 0 -e 20  -train_mode acc -log_folder test -log test_Full_batch -save_cp b0_100e -full_batch  -patience 2  -lr 1e-3`

* **Training DEGC**

> DEGC+Finetune:
> `python -u run_DEGC.py -d Taobao2014 -bm MGCCF -alg DEGC+Finetune -de 0 -e 25 -train_mode sep -log_folder test -log  test_DEGC+Finetune -save_cp b0_100e  -patience 2 -lr 1e-3 `
>
> DEGC+LWC-KD:
> `python -u run_DEGC.py -d Taobao2014 -bm MGCCF -alg DEGC+LWC_KD -de 0 -e 25  -train_mode sep -log_folder test -log test_DEGC+LWC_KD -save_cp b0_100e -layer_wise 1 -contrastive_mode 'Multi' -lambda_contrastive 100,100,1000 -con_positive 15 -con_ratios 2,1,2,1,1,1,1 -patience 2 -lr 1e-3  `
>
