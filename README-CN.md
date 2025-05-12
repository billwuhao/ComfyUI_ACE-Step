[中文](README-CN.md)|[English](README.md)

# ACE-Step 的 ComfyUI 节点

快速, 高质量音乐生成, "重绘", Remix, 编辑, 扩展等, Windows, Linux, Mac 应该都支持(未做完整测试).

## 📣 更新

[2025-05-12]⚒️: 增加模型加载节点, 可自定义模型加载. 感谢 @[thezveroboy](https://github.com/thezveroboy) 的贡献. 增加 `cpu_offload`, 8g 显存可用, 速度更快.

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-12_09-37-42.png)

[2025-05-10]⚒️: 增加歌词语言转换节点, 为 [ComfyUI](https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1) 官方版 ACE-Step 工作流提供多语言支持. 

[2025-05-07]⚒️: 发布版本 v1.0.0. 

## 使用

增加了多语言转换节点, 为 [ComfyUI](URL_ADDRESS.comfy.org/tutorials/audio/ace-step/ace-step-v1) 官方版 ACE-Step 工作流提供多语言支持. 目前，ACE Step 支持 19 种语言，但以下 10 种语言有更好的支持：
- 英语：[en]
- 中文：[zh]
- 俄文：[ru]
- 西班牙文：[es]
- 日文：[ja]
- 德文：[de]
- 法文：[fr]
- 葡萄牙文：[pt]
- 意大利文：[it]
- 韩文：[ko]

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-10_19-26-46.png)

- 生成:

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-07_19-53-51.png)

- "重绘":

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-07_19-59-22.png)

- 扩展:

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-07_20-04-02.png)

- 编辑:

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-07_20-09-52.png)

- 自动生成歌词, prompt, 暂停工作流, 修改然后点击 `continue workflow` 继续工作流 [example](workflow-examples/ACE-gen-automated-composition.json). 可用最新的 Gemini, Qwen3, 以及 DeepSeek v3.:

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-11_00-38-33.png)

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_ACE-Step.git
cd ComfyUI_ACE-Step
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B

模型手动下载放到 `models\TTS\ACE-Step-v1-3.5B` 目录下, 结构如下:

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

## 鸣谢

[ACE-Step](https://github.com/ace-step/ACE-Step)