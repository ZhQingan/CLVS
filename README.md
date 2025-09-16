### Under Construction ...

---

# Cross-Layer Vision Smoothing: Enhancing Visual Understanding via Sustained Focus on Key Objects in Large Vision-Language Models

We propose Cross-Layer Vision Smoothing, a method that mitigates the layer-by-layer decay of visual attention to enable LVLMs to maintain sustained focus on key objects in the image, thereby enhancing their understanding of these objects.

---

## Setup

### Environment

```bash
conda create -n clvs -y python=3.10
conda activate clvs

pip install -r requirements.txt
```

**Note**: The dependencies are referred to LLaVA-v1.5. For LLaVA-Next and Qwen2.5-VL-Instruct, you can also easily set up the environment by following the instructions from their official repositories.

### Datasets

All benchmarks need to be processed into structurally consistent JSON files.

Some samples could be found in `data/samples.json`.


## Evaluate CLVS


### Quick start

We developed a shell script `scripts/eval_all.sh` that can execute benchmarks end-to-end.

**Inference**

You can also test your own dataset.
```bash
model_name=model_name
python ./eval/object_hallucination_vqa_${model_name}.py \
--model-path /path/to/model \
--question-file /path/to/test.json \
--image-folder /path \
--answers-file /path/to/results.json

```

**Evaluation**

```bash
# POPE and R-Bench
python eval.py --ans_folder path/to/results.json
```
AMBER and MME has its own evaluation method; please refer the official repositories for result evaluation:  [AMBER](https://github.com/junyangwang0410/AMBER), [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)