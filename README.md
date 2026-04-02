# CheXOne: A Reasoning-Enabled Vision–Language Foundation Model for Chest X-ray Interpretation

<p align="center">
    <br>
    <img src="asset/chexone_logo1.png"/>
    <br>
<p>
<p align="center">
<!-- <a href="https://modelscope.cn/home">ModelScope Community Website</a>
<br>
        <a href="README_CN.md">中文</a> &nbsp ｜ &nbsp English &nbsp
</p> -->

<p align="center">
<img src="https://img.shields.io/badge/python-3.10-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A52.7-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.19-5D91D4.svg"></a>
<a href="https://github.com/modelscope/ms-swift/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/ms-swift"></a> 
<a href="https://huggingface.co/StanfordAIMI/CheXOne">
  <img src="https://img.shields.io/badge/dynamic/json?longCache=true&style=flat&label=Downloads&query=downloads&url=https://huggingface.co/api/models/StanfordAIMI/CheXOne&color=yellow">
</a>
</p>



<p align="center">
    <a href="https://arxiv.org/pdf/2604.00493v1">Paper</a> &nbsp ｜ &nbsp <a href="https://huggingface.co/StanfordAIMI/CheXOne">Hugging Face</a> &nbsp ｜ &nbsp <a href="https://rexrank.ai/">ReXRank Leaderboard</a>
</p>

## 📖 Table of Contents
- [Introduction](#-introduction)
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-Start)
- [Data](#-Data)
- [Train](#-Train)
- [Inference](#-Inference)
- [User Study](#-User~Study)
- [License](#-License)
- [Citation](#-citation)




## 📝 Introduction

CheXOne is a reasoning-aware vision-language model for chest X-ray interpretation.

✨ **Key Features:**
- **Reasoning Capability:** Generates explicit reasoning traces alongside final answers.
- **Multi-Task Support:** Handles Visual Question Answering (VQA), report generation, and visual grounding tasks.
- **Resident-Level Reporting:** Achieves report quality that matches or surpasses resident-written reports in 50+% of studied cases.
- **Two Inference Modes:**
  - **Reasoning Mode:** Higher performance with explicit reasoning traces.
  - **Instruct Mode:** Faster inference without reasoning traces.

**This code release includes:**
- Step-by-step instructions to reproduce our methodology.
- Data preparation scripts for CheXInstruct-v2 and CheXReason.
- Complete training code, including instruction tuning and GRPO.
- Complete inference code, with evaluation for our model and comparative baselines.
- User study scripts and related documentation.
- Code for generating publication figures.

## 🛠️ Installation
```shell
https://github.com/YBZh/CheXOne.git
cd CheXOne
pip install -e .
```

Training and Fast Inference Environment:

|              | Range        | Recommended         | Notes                                     |
|--------------|--------------|---------------------|-------------------------------------------|
| python       | >=3.10,<3.12 | 3.10/3.11           |                                           |
| cuda         | 12.x         | cuda12              | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0        | 2.7.1               |                                           |
| transformers | >=4.33       | 4.56.2              |                                           |
| modelscope   | >=1.23       | 1.30.0              |                                           |
| peft         | >=0.11,<0.18 | 0.17.1              |                                           |
| flash_attn   |              | 2.5.8               |                                           |
| trl          | >=0.15,<0.21 | 0.20.0              | RLHF                                      |
| deepspeed    | >=0.14       | 0.17.6              | Training                                  |
| vllm         | >=0.5.1      | 0.10.1.1            | Inference/Deployment                      |
| sglang       | >=0.4.6      | 0.4.10.post2         | Inference/Deployment                      |
| lmdeploy     | >=0.5        | 0.10.1              | Inference/Deployment                      |
| evalscope    | >=1.0        | 1.0.2               | Evaluation                                |
| gradio       |              | 5.32.1              | Web-UI/App                                |

For more optional dependencies, you can refer to [here](https://github.com/YBZH/CheXOne/examples/train/chexone/requirements.txt).


## 🚀 Quick Start

CheXOne is post-trained on the Qwen2.5VL-3B-Instruct model, which is integrated in the latest HuggingFace Transformers. We advise you to build `transformers` from source as follows:

```shell
pip install git+https://github.com/huggingface/transformers accelerate
```

Below is an example usage to get started:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "StanfordAIMI/CheXOne", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "StanfordAIMI/CheXOne",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# Default processor
processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXOne")

# The default range for the number of visual tokens per image in the model is 4-16384.
# We recommend to set max_pixels=512*512 to align with the training setting.
# min_pixels = 256*28*28
# max_pixels = 512*512
# processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXOne", min_pixels=min_pixels, max_pixels=max_pixels)

# Inference Mode: Reasoning
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://raw.githubusercontent.com/YBZh/CheXOne/main/asset/cxr.jpg",
            },
            {
                "type": "text",
                "text": "Write an example findings section for the CXR. Please reason step by step, and put your final answer within \\boxed{{}}.",
            },
        ],
    }
]

# Inference Mode: Instruct
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://raw.githubusercontent.com/YBZh/CheXOne/main/asset/cxr.jpg",
#             },
#             {
#                 "type": "text",
#                 "text": "Write an example findings section for the CXR.",
#             },
#         ],
#     }
# ]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)
```

## 📂 Data
### CheXinstruct-v2 and CheXReason (<span style="color:red">Under preparation and policy checking</span>)

- **Images**: Please download the CXR images following the instructions in the corresponding datasets: [📂 data.md](.data.md)
- **Texts**: 
  - [📊 CheXinstruct-v2](链接A)
  - [🧠 CheXReason](链接B)

## 🏋️ Train

### 1. Instruction Tuning

See: [`examples/train/chexone/train_script/1_sft.sh`](examples/train/chexone/train_script/1_sft.sh)

This step performs supervised fine-tuning using curated CheXinstruct-v2 and CheXReason.

---

### 2. Low Variance Filtering for GRPO (<span style="color:red">Under preparation</span>)

See: [`examples/train/chexone/grpo_prepare`](examples/train/chexone/grpo_prepare)

To ensure strong learning signals for GRPO, we filter out low-variance samples. For each candidate, several stochastic model runs are used to estimate reward variance, and only the high informative samples in each category—those with highest reward variance—are selected. This strategy improves GRPO effectiveness and efficiency.

---

### 3. GRPO Training

See: [`examples/train/chexone/train_script/2_grpo.sh`](examples/train/chexone/train_script/2_grpo.sh)

This step further optimizes the model with the GRPO algorithm to improve reasoning capabilities and robustness.


## 🧪 Inference

### 1. Normal Inference as stated in [Quick Start](#-quick-Start)

### 2. Fast Inference with vLLM

See: [`examples/train/chexone/prepare_testing/3_inference_code/CheXOne.sh`](examples/train/chexone/prepare_testing/3_inference_code/CheXOne.sh)

### 3. Inference of Other Models

See: [`examples/train/chexone/prepare_testing/3_inference_code/OtherModels`](examples/train/chexone/prepare_testing/3_inference_code/OtherModels)

## 👩‍⚕️ Reader Study
See: [`examples/train/chexone/Reader-Study`](examples/train/chexone/Reader-Study)


## 🏛 License

This framework is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE). For models and datasets, please refer to the original resource page and follow the corresponding License.

## 📎 Citation

```bibtex
@misc{zhang2026reasoningenabledvisionlanguagefoundationmodel,
      title={A Reasoning-Enabled Vision-Language Foundation Model for Chest X-ray Interpretation}, 
      author={Yabin Zhang and Chong Wang and Yunhe Gao and Jiaming Liu and Maya Varma and Justin Xu and Sophie Ostmeier and Jin Long and Sergios Gatidis and Seena Dehkharghani and Arne Michalson and Eun Kyoung Hong and Christian Bluethgen and Haiwei Henry Guo and Alexander Victor Ortiz and Stephan Altmayer and Sandhya Bodapati and Joseph David Janizek and Ken Chang and Jean-Benoit Delbrouck and Akshay S. Chaudhari and Curtis P. Langlotz},
      year={2026},
      eprint={2604.00493},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.00493}, 
}
