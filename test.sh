# Testing the Synapse dataset using a weight trained on 18 training samples
CUDA_VISIBLE_DEVICES=0 python test_Synapse.py --split train18 --is_savenii
# Testing the Synapse dataset using a weight trained on 9 training samples
CUDA_VISIBLE_DEVICES=0 python test_Synapse.py --split train9 --is_savenii
# Testing the Synapse dataset using a weight trained on 6 training samples
CUDA_VISIBLE_DEVICES=0 python test_Synapse.py --split train6 --is_savenii


# Testing the ACDC dataset using a weight trained on 70 training samples
CUDA_VISIBLE_DEVICES=0 python test_ACDC.py --train_num 70 --is_savenii
# Testing the ACDC dataset using a weight trained on 28 training samples
CUDA_VISIBLE_DEVICES=0 python test_ACDC.py --train_num 28 --is_savenii
# Testing the ACDC dataset using a weight trained on 7 training samples
CUDA_VISIBLE_DEVICES=0 python test_ACDC.py --train_num 7 --is_savenii