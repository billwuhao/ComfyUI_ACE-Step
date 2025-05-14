[中文](README-CN.md)|[English](README.md)

# ACE-Step Nodes for ComfyUI

Fast, high-quality music generation, "repainting", remixing, editing, extending, and more.  Windows, Linux, and Mac should all be supported (not fully tested).

## 📣 Updates

[2025-05-14]⚒️: 

- Support Lora, [ACE-Step-v1-chinese-rap-LoRA](https://huggingface.co/ACE-Step/ACE-Step-v1-chinese-rap-LoRA) download and place it in the `models\TTS\ACE-Step-v1-3.5B\loras` directory.
```
        loras
        └─ACE-Step-v1-chinese-rap-LoRA
                config.json
                pytorch_lora_weights.safetensors
```
- Add many delicious reference song parameters.

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-14_14-23-50.png)

[2025-05-12]⚒️: Add model loader node, customizable model loading. Thank you for @[thezveroboy's](https://github.com/thezveroboy) contribution. Add `cpu_offload`, 8g of memory available, faster speed.

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-12_09-37-42.png)

[2025-05-10]⚒️: Added a node for lyrics language conversion to provide multilingual support for the official [ComfyUI](https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1) ACE-Step workflow.

[2025-05-07]⚒️: Released version v1.0.0.

## Usage

- Lora:

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-14_14-10-23.png)

Added a node for lyrics language conversion to provide multilingual support for the official [ComfyUI](https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1) ACE-Step workflow. Currently, ACE-Step supports 19 languages, but the following ten languages have better support:
- English
- Chinese: [zh]
- Russian: [ru]
- Spanish: [es]
- Japanese: [ja]
- German: [de]
- French: [fr]
- Portuguese: [pt]
- Italian: [it]
- Korean: [ko]

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-10_19-26-46.png)

- Generation:

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-07_19-53-51.png)

- "Repainting":

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-07_19-59-22.png)

- Extending:

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-07_20-04-02.png)

- Editing:

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-07_20-09-52.png)

- Automatically generate lyrics, prompt, pause workflow, modify and then click `continue workflow` to continue workflow [example](workflow-examples/ACE-gen-automated-composition.json). The latest Gemini, Qwen3, and DeepSeek v3 are available:

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-11_00-38-33.png)

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_ACE-Step.git
cd ComfyUI_ACE-Step
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B

Manually download the model and place it in the `models\TTS\ACE-Step-v1-3.5B` directory, with the following structure: 

```
ACE-Step-v1-3.5B
│
├─ace_step_transformer
│      config.json
│      diffusion_pytorch_model.safetensors
│
├─music_dcae_f8c8
│      config.json
│      diffusion_pytorch_model.safetensors
│
├─music_vocoder
│      config.json
│      diffusion_pytorch_model.safetensors
│
└─umt5-base
        config.json
        model.safetensors
        special_tokens_map.json
        tokenizer.json
        tokenizer_config.json
```

## Acknowledgements

[ACE-Step](https://github.com/ace-step/ACE-Step)
