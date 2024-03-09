# Training the Synapse dataset for 18 sample
CUDA_VISIBLE_DEVICES=0,1 python train_Synapse.py --split train18
# Training the Synapse dataset for 9 sample
CUDA_VISIBLE_DEVICES=0,1 python train_Synapse.py --split train9
# Training the Synapse dataset for 6 sample
CUDA_VISIBLE_DEVICES=0,1 python train_Synapse.py --split train6

# Training the ACDC dataset for 70 sample
CUDA_VISIBLE_DEVICES=0,1 python train_ACDC.py --train_num 70
# Training the ACDC dataset for 28 sample
CUDA_VISIBLE_DEVICES=0,1 python train_ACDC.py --train_num 28
# Training the ACDC dataset for 7 sample
CUDA_VISIBLE_DEVICES=0,1 python train_ACDC.py --train_num 7