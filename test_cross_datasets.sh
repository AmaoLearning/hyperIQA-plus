CUDA_VISIBLE_DEVICES=6 python test_IQA.py --model_path ./checkpoints --model_name hyperIQA_residual_v2LossTest_l2 --datasets koniq-10k spaq kadid agiqa
CUDA_VISIBLE_DEVICES=7 python test_IQA.py --model_path ./checkpoints --model_name hyperIQA_residual_v2Loss_srcc --datasets koniq-10k spaq kadid agiqa
CUDA_VISIBLE_DEVICES=0 python test_IQA.py --model_path ./checkpoints --model_name hyperIQA_residual_v2Loss_plcc --datasets koniq-10k spaq kadid agiqa
CUDA_VISIBLE_DEVICES=2 python test_IQA.py --model_path ./checkpoints --model_name hyperIQA_residual_v2Loss_rank --datasets koniq-10k spaq kadid agiqa
CUDA_VISIBLE_DEVICES=5 python test_IQA.py --model_path ./checkpoints --model_name hyperIQA_residual_v2Loss_pairwise --datasets koniq-10k spaq kadid agiqa