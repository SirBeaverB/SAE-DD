 #!/bin/bash

# Install dependencies
# pip install -r requirements.txt

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi

# Run fine-tuning with multi-GPU support
python finetune_gpt2.py \
    --data_file "csqa_naive_distill/pure_sentences/pure_top_70%_sentences.json" \
    --output_dir "finetune/csqa70%" \
    --model_name "gpt2" \
    --num_epochs 3 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --max_length 512 \
    --use_multi_gpu \
    --gradient_accumulation_steps 8 