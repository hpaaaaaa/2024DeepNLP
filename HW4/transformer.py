from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import torch

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_train=False
# 加载预训练的GPT-2模型和分词器
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 添加填充token
tokenizer.pad_token = tokenizer.eos_token

# 加载并准备数据集
def load_novel_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return Dataset.from_dict({'text': [text]})

dataset = load_novel_data('天龙八部train.txt')

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512,add_special_tokens=True)

def generate_novel_continuation(prompt, max_length=200, num_return_sequences=1):
    # 编码输入文本
    inputs = fine_tuned_tokenizer.encode(prompt, return_tensors='pt').to(device)

    # 使用模型生成文本
    outputs = fine_tuned_model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True
    )

    # 解码并返回生成的文本
    continuations = [fine_tuned_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return continuations



if __name__ == '__main__':
    if is_train:
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

        # 数据整理器，用于语言模型训练
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # GPT-2 是自回归模型，不使用掩码语言模型（MLM）
        )

        # 设置训练参数
        training_args = TrainingArguments(
            output_dir='./results',
            overwrite_output_dir=True,
            num_train_epochs=200,
            per_device_train_batch_size=1,
            save_steps=10_000,
            save_total_limit=2,
            fp16=True,  # 启用混合精度训练
            learning_rate=5e-5  # 调整学习率
        )

        # 定义Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )

        # 微调模型
        trainer.train()

        # 保存微调后的模型
        model.save_pretrained('./fine_tuned_model')
        tokenizer.save_pretrained('./fine_tuned_model')
    else:
        # 加载微调后的模型和分词器
        fine_tuned_model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model').to(device)
        fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')
        # 示例输入
        prompt = "众人相顾愕然，没料想皇帝一句话不说，一口酒不饮，竟便算赴过了酒宴。"

        # 生成小说续写
        continuations = generate_novel_continuation(prompt, max_length=200, num_return_sequences=1)

        # 输出结果
        for i, continuation in enumerate(continuations):
            print(f"续写 {i + 1}:\n{continuation}\n")