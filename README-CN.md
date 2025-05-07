[中文](README-CN.md)|[English](README.md)

# ACE-Step 的 ComfyUI 节点

快速, 高质量音乐生成, "重绘", Remix, 编辑, 扩展等, Windows, Linux, Mac 应该都支持(未做完整测试).

示例:

- 生成:

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-07_19-53-51.png)

- "重绘":

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-07_19-59-22.png)

- 扩展:

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-07_20-04-02.png)

- 编辑:

![](https://github.com/billwuhao/ComfyUI_ACE-Step/blob/main/images/2025-05-07_20-09-52.png)

## 📣 更新

[2025-05-07]⚒️: 发布版本 v1.0.0. 

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

模型会自动下载到 `models\TTS\ACE-Step-v1-3.5B` 目录下, 也可以手动下载放到该目录下, 手动下载结构如下:

https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B

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