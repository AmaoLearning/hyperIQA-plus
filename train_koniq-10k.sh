# train residual net
#CUDA_VISIBLE_DEVICES=6 python train_test_IQA.py --dataset koniq-10k --model_name hyperIQA_residual_v2 --model_type residual --test_batch_size 96

# test one train and test process
#CUDA_VISIBLE_DEVICES=6 python train_test_IQA.py --dataset koniq-10k --model_name hyperIQA_residual_v2LossTest --model_type residual --epochs 1 --train_test_num 1 --test_batch_size 96 --loss_type l1
CUDA_VISIBLE_DEVICES=6 python train_test_IQA.py --dataset koniq-10k --model_name hyperIQA_residual_v2Loss --model_type residual --test_batch_size 96 --loss_type l2
CUDA_VISIBLE_DEVICES=7 python train_test_IQA.py --dataset koniq-10k --model_name hyperIQA_residual_v2Loss --model_type residual --test_batch_size 96 --loss_type srcc
CUDA_VISIBLE_DEVICES=0 python train_test_IQA.py --dataset koniq-10k --model_name hyperIQA_residual_v2Loss --model_type residual --test_batch_size 96 --loss_type plcc
CUDA_VISIBLE_DEVICES=2 python train_test_IQA.py --dataset koniq-10k --model_name hyperIQA_residual_v2Loss --model_type residual --test_batch_size 96 --loss_type rank
CUDA_VISIBLE_DEVICES=5 python train_test_IQA.py --dataset koniq-10k --model_name hyperIQA_residual_v2Loss --model_type residual --test_batch_size 96 --loss_type pairwise

# test simple epoch running time
#CUDA_VISIBLE_DEVICES=6 python train_test_IQA.py --dataset koniq-10k --model_name hyperIQA_baseline_ac --model_type baseline --epochs 1 --train_test_num 1 --test_batch_size 96
