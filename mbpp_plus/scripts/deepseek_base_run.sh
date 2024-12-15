SEMCODER_DIR="/home/preethiprakash2001/SemCoderBreakers/mbpp_plus/perturbed-datasets/partial"
VLLM_DIR="output_dir/deepseek_base"

folders=("natgen" "func_name" "format", "nlaugmenter)

for folder in "${folders[@]}"; do
    for file in "$SEMCODER_DIR/$folder"/*.jsonl; do
        filename=$(basename "$file")
        command="MBPP_OVERRIDE_PATH=\"$file\" evalplus.codegen --model "deepseek-ai/deepseek-coder-6.7b-base" --greedy --dataset mbpp --backend vllm --root  $VLLM_DIR/$folder" 
        echo "Executing: $command"
        eval "$command"

        # Clear CUDA cache using Python
        echo "Clearing CUDA cache..."
        python3 - <<EOF
import torch
print("Clearing CUDA cache...")
torch.cuda.empty_cache()
torch.cuda.synchronize()
print("CUDA cache cleared successfully.")
EOF

    done
done
