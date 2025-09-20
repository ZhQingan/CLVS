{
export CUDA_VISIBLE_DEVICES='0'
model_name='llava'

seed=17
max_len=1
gamma=0.8
beta=0.8
delta=0.5
use_add=true

if [[ $model_name == "llava" ]]; then
  model_path="/path/to/models/llava-v1.5-7b"
elif [[ $model_name == "llava-13b" ]]; then
  model_path="/path/to/models/llava-v1.5-13b"
elif [[ $model_name == "qwen25-vl" ]]; then
  model_path="/path/to/models/Qwen2.5-VL-7B-Instruct"
elif [[ $model_name == "llava-next" ]]; then
  model_path="/path/to/models/llama3-llava-next-8b"
else
  model_path=""
fi

image_folder=/path/to/benchmark/images
dataset='samples'
python ./eval/object_hallucination_vqa_${model_name}.py \
--model-path $model_path \
--question-file ../data/${dataset}.json \
--image-folder $image_folder \
--answers-file ./results/${model_name}.${dataset}.greedy.seed${seed}.jsonl \
--use_add $use_add \
--max_gen_len 1 \
--gamma $gamma \
--beta $beta \
--delta $delta \
--seed $seed

}

