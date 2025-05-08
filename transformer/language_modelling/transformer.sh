#! /bin/bash
#SBATCH --partition=long

# source /home/dmerkx/torch_env/bin/activate

# export CUDA_VISIBLE_DEVICES=1

# python -u nwp_tf.py -model_ids 1 -results_loc /Users/jairiley/Desktop/BOW_Ngrams/transformer/pos > tf1_.out &
# sleep 1
# python -u nwp_tf.py -model_ids 2 -batch_size 10 -results_loc /Users/jairiley/Desktop/BOW_Ngrams/transformer/pos > tf2_.out 
# # sleep 1
# python -u nwp_tf.py -model_ids 3 -batch_size 20 -results_loc /Users/jairiley/Desktop/BOW_Ngrams/transformer/pos > tf3_.out 
# # sleep 1
# python -u nwp_tf.py -model_ids 4 -batch_size 40 -results_loc /Users/jairiley/Desktop/BOW_Ngrams/transformer/pos > tf4_.out 
# # sleep 1
# python -u nwp_tf.py -model_ids 8 -batch_size 5 -lr 0.025 -results_loc /Users/jairiley/Desktop/BOW_Ngrams/transformer/pos > tf8_.out 
# python -u nwp_tf.py -model_ids 5 -batch_size 10 -lr 0.025 -results_loc /Users/jairiley/Desktop/BOW_Ngrams/transformer/pos > tf5_.out 
# # sleep 1
python -u nwp_tf.py -model_ids 2 -batch_size 20 -lr 0.025 -results_loc /Users/jairiley/Desktop/BOW_Ngrams/transformer/pos > tf6_.out 
# sleep 1
# python -u nwp_tf.py -model_ids 7 -batch_size 40 -lr 0.025 -results_loc /Users/jairiley/Desktop/BOW_Ngrams/transformer/pos > tf7_.out 
# # sleep 1
