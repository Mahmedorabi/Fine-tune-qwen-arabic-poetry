# Arabic Poetry AI — Fine-tuning Qwen3-1.7B with LoRA

This project demonstrates how to fine-tune the [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) model on a synthetic Arabic poetry dataset using **LoRA (Low-Rank Adaptation)**. The result is an efficient Arabic-language model capable of generating poetic responses to user prompts.

---

## 🚀 Project Highlights

* 🔤 Language: **Arabic**
* 🧠 Base Model: [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B)
* 🧩 Fine-tuning Technique: **LoRA (r=8, alpha=16)**
* ⏱️ Training Time: \~24 minutes on Google Colab L4 GPU
* 📦 Output Model: [Hugging Face Repo](https://huggingface.co/mohammed-orabi2/qwen-poetry-arabic-lora)

---

## 📁 Project Structure

```
├── arabic_poetry_1000.json       # Synthetic Arabic poetry dataset
├── requirements.txt              # Project dependencies
├── train_lora.py                 # Full training & deployment script
└── README.md                     # You're here
```

---

## 🛠 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📚 Dataset

The dataset consists of 1000 examples of user prompts in Arabic with corresponding poetic responses. All formatted in conversational style (role: user/assistant) to match chat model expectations.

---

## 🧪 Training Workflow

```python
# 1. Load & format data
# 2. Apply chat template using Qwen tokenizer
# 3. Tokenize, pad, and set labels for CausalLM
# 4. Apply LoRA (q_proj, v_proj)
# 5. Train for 5 epochs using Hugging Face Trainer
```

Key libraries used:

* `transformers`
* `datasets`
* `peft`
* `huggingface_hub`

---

## 🧾 Example Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
model = PeftModel.from_pretrained(base, "mohammed-orabi2/qwen-poetry-lora2")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

prompt = "اكتب لي بيت شعر عن النجاح."
chat = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## ☁️ Deploy to Hugging Face

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="qwen-poetry-lora2",
    repo_id="mohammed-orabi2/qwen-poetry-lora2",
    repo_type="model"
)
```

---

## 📈 Results

* 90% of generated responses followed poetic structure and matched topics.
* Excellent performance for creative and expressive Arabic content.

---

## 📌 Notes

* The model is not intended for factual tasks or safety-critical applications.
* Designed specifically for educational and creative purposes.

---

## 👤 Author

**Mohammed Orabi**
[Hugging Face](https://huggingface.co/mohammed-orabi2)

---

## 📄 License

Apache 2.0 (inherited from Qwen3-1.7B)

---

## 💬 Feedback & Contributions

Pull requests and issues are welcome! Let’s collaborate to expand Arabic LLM applications.
