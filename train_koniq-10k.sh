# train residual net
#CUDA_VISIBLE_DEVICES=6 python train_test_IQA.py --dataset koniq-10k --model_name hyperIQA_residual_v2 --model_type residual --test_batch_size 96

# test one train and test process
CUDA_VISIBLE_DEVICES=6 python train_test_IQA.py --dataset koniq-10k --model_name hyperIQA_residual_v2_test --model_type residual --train_test_num 1 --test_batch_size 96

# test simple epoch running time
#CUDA_VISIBLE_DEVICES=6 python train_test_IQA.py --dataset koniq-10k --model_name hyperIQA_baseline_ac --model_type baseline --epochs 1 --train_test_num 1 --test_batch_size 96
