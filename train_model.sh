torchrun --standalone --nproc_per_node=2 train.py \
--train-path <PATH TO TOKENIZED TRAINING DATASET> \
--dev-path /home/shared/paloma_c4_100_domains_val_tokenized.bin \
--output-dir <PATH TO SAVE FINAL MODEL> \
--vocab-size 50257 \
--context-length 512 \
--d-model 768 \
--num-layers 12 \
--num-heads 12 \
--d-ff 3072 \
--attn-pdrop 0.1 \
--residual-pdrop 0.1 \
--batch-size 128 \
--train-steps 200000 \
--eval-iters 1000 \
--eval-interval 2000 \
--learning-rate 1e-3 \
--lr-scheduler cosine \
--weight-decay 0.1 \
--warmup-ratio 0.01 \
--grad-clip 1.0 \
--dtype bfloat16 \
--wandb-project <INSERT WANDB PROJECT NAME> \
--compile