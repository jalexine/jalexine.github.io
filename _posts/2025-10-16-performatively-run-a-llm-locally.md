---
layout: post
title: "performatively run a llm locally"
category: example
---

<span style="color:white"> Want to run performatively a LLM on your phone just like me?

Here’s how..</span>


<p align="center">
  <img src="assets/mattisdoviereyes.gif" alt="mattisdovier" width="500" style="height:auto;" />
</p>
*Mattis Dovier*


## <span style="color: white;">1. Device </span>
  

If you’re on **Android**, start by downloading **Termux** from [F-Droid](https://f-droid.org/en/packages/com.termux/).  
If iOS, I guess [iSHell](https://apps.apple.com/us/app/ish-shell/id1436902243).  

Termux basically gives you a Linux terminal on your phone. It’s where you’ll run commands, clone repositories, and compile the model.  
Once it’s installed, you can already run smaller models, like **Phi-3 mini** or **Mistral 7B quantized**, directly on your device.  

Don’t expect full **gpt4** power. The goal here isn’t to replace ChatGPT, it’s just to experiment, play...  
You’ll see the model responding in your terminal, offline, with zero cloud connection.  

<br>

<span style="color:white"> If you’re on a **PC or laptop**, everything is also possible here.  
You can follow this performative article to get started, but if you want stable performance and efficiency, follow a serious setup ressource </span>

Your hardware decides everything: RAM, VRAM, and whether your GPU supports CUDA or Metal, that’s what defines how fast your model will run and if it can run..
You can ask ChatGPT if your device supports the model you want to use.  

<br>

## <span style="color: white;">2. The Model </span>
 

Models come in different forms, and for local inference you need the **quantized** kind — files ending with `.gguf`.  
These are compressed versions that can actually fit into your hardware without exploding your RAM.  

You can find models, including uncensored ones, on [Hugging Face](https://huggingface.co/models) or the [llama.cpp community repos](https://github.com/ggerganov/llama.cpp/discussions).  
Once downloaded, create a folder called `models` inside your `llama.cpp` directory and drop the `.gguf` file there.  

<br>


<img src="assets/yudho.gif" alt="morales" width="320" />
* [Yudho](https://yudho.xyz/) *


## <span style="color: white;">3. Installation  

I’m using **llama.cpp** because it’s the simplest and most flexible option.  
If you want to try other setups, you can check **Ollama**, **LM Studio**, **text-generation-webui**, and others.  

### On Termux / Android  

Install packages with:  
```bash
pkg install git cmake clang make
```

Then clone and build:
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build -j
```
<br>
## 4. Run the Model

### Basic command
<pre style="color:#9aa0a6;opacity:.9">./build/bin/llama-cli -m ./models/llama3.1-8b-instruct.Q4_K_M.gguf -p "hey do you have weed?"</pre>
This runs a one-off prompt and prints the reply.

### Interactive session
<pre style="color:#9aa0a6;opacity:.9">./build/bin/llama-cli -m ./models/lexi-llama-3-8B-Uncensored-4Q_K_M.gguf -t 8 -c 2048 --temp 0.7 --interactive --color</pre>
`t` = threads, `c` = context length, `--temp` = creativity, `--interactive` = chat mode, `--color` = colored output.  
Type your questions, hit Enter; replies are in real time, fully offline.

### Give it a personality
Before your first message, set a system prompt:
<pre style="color:#9aa0a6;opacity:.9">[SYSTEM PROMPT: You are my everyday assistant and boyfriend. Personality: witty, a bit rebellious, playful but sharp. You give me practical advice, creative ideas, and honest takes even if it’s blunt. You hate advertisements. Never corporate, never boring.]</pre>
Paste it in `models/sys.txt` (or as the first line in the session).


if you have any questions about something that isn’t working here, please ask chatgpt.

<br>

<br>
will add ressources here soon.

