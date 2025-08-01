OUTPUT=$1
MAX_LEN=$2
MAX_STEPS=$3
W_SZ=$4
POOL_SZ=$5
POOL_FUNC=$6

if [ "$OUTPUT" == "" ]; then
    OUTPUT="output/debug"
fi

if [ "$MAX_LEN" == "" ]; then
    MAX_LEN=8192
fi

if [ "$MAX_STEPS" == "" ]; then
    MAX_STEPS=1000
fi

if [ "$W_SZ" == "" ]; then
    W_SZ=16
fi

if [ "$POOL_SZ" == "" ]; then
    POOL_SZ=16
fi

if [ "$POOL_FUNC" == "" ]; then
    POOL_FUNC="cca"
fi

mkdir -p $OUTPUT
echo "OUTPUT=$OUTPUT"
echo "MAX_LEN=$MAX_LEN"
echo "MAX_STEPS=$MAX_STEPS"
echo "W_SZ=$W_SZ"
echo "POOL_SZ=$POOL_SZ"
echo "POOL_FUNC=$POOL_FUNC"

mkdir -p $OUTPUT

torchrun --nnodes=1 --nproc_per_node=8 fine-tune.py  \
        --model_name_or_path /path/to/model \
		--rope_theta 5e5 \
        --bf16 True \
        --output_dir $OUTPUT \
		--logging_dir $OUTPUT \
        --data_dir /path/to/traning_data \
        --model_max_length $MAX_LEN \
        --replace True \
		--pool_func $POOL_FUNC \
		--window_size $W_SZ \
		--pool_size $POOL_SZ \
		--only_attn False \
        --num_train_epochs 1  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 4     \
        --evaluation_strategy "steps"     \
		--eval_steps 100000 \
        --save_strategy "steps"     \
        --save_steps 250     \
        --save_total_limit 10     \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
		--max_grad_norm 2.0 \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
		--log_level "debug" \
        --deepspeed "ds_configs/stage3_fast.json" \
        --max_steps $MAX_STEPS 2>&1 | tee $OUTPUT/training.log