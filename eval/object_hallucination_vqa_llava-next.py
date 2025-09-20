import argparse
import torch
import os
import json
from tqdm import tqdm
import random
# import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava_next.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_next.conversation import conv_templates, SeparatorStyle
from llava_next.model.builder import load_pretrained_model
from llava_next.utils import disable_torch_init
from llava_next.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images, KeywordsStoppingCriteria

from PIL import Image
import math

# import kornia
from transformers import set_seed

import copy

tpn_map={
    'fp16': 'float16',
    'fp32': 'float32',
    'bf16': 'bfloat16',
}
def eval_model(args):
    # Model
    disable_torch_init()
    answers_file = os.path.expanduser(args.answers_file)
    if os.path.exists(answers_file):
        print(f"File existed: {answers_file}")
        exit()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tpn = tpn_map[args.torch_type]
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map='cuda', attn_implementation='eager', torch_dtype=tpn)
    model.eval()
    model.model.lm_head = model.lm_head
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):

        image_file = line["image"]
        idx = line['question_id']
        qs = line["question"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # conv = conv_templates[args.conv_mode].copy()
        conv = copy.deepcopy(conv_templates[args.conv_mode])
        conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        prompt += '\n\n'
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_sizes = [image.size]
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep
        keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                user_len=input_ids.size(-1)-45-1, #sys_len+img_token
                images=image_tensor.unsqueeze(0).to(model.dtype).cuda(),
                image_sizes=image_sizes,
                gamma=args.gamma,
                beta=args.beta,
                delta=args.delta,
                do_sample=False,
                use_add=args.use_add,
                max_new_tokens=args.max_gen_len,
                min_new_tokens=1,
                use_cache=True,
                eos_token_id=tokenizer.convert_tokens_to_ids(stop_str),
                pad_token_id=tokenizer.convert_tokens_to_ids(stop_str),
                )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        ans_file.write(json.dumps({"question_id": line['question_id'],
                                   "question": line['question'],
                                   "output": outputs,
                                   "label": line['label'],
                                   "prompt": prompt,
                                   "model_id": model_name,
                                   "image": image_file,
                                   }) + "\n")
        ans_file.flush()
    ans_file.write(json.dumps(vars(args)) + '\n')
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_llama_3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--use_add", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--delta", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_type", type=str, default="bf16", choices=['fp16','fp32','bf16'])

    args = parser.parse_args()
    set_seed(args.seed)

    print(str(args), flush=True)
    eval_model(args)
