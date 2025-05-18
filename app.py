import gradio as gr
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load Chinese BERT-Large model fine-tuned for QA
model_name = "cgt/Roberta-wwm-ext-large-qa"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# QA function
def answer_question(context, question):
    try:
        inputs = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]

        with torch.no_grad():
            outputs = model(**inputs)

        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1

        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx])
        )

        return answer.strip() if answer.strip() else "⚠️ 抱歉，我无法从上下文中找到答案。"

    except Exception as e:
        return f"❌ 错误：{str(e)}"

# 创建 Gradio 界面
with gr.Blocks(title="中文BERT问答系统") as demo:
    gr.Markdown("## 请在下方分别输入上下文和问题。")

    with gr.Row():
        context_input = gr.Textbox(label="📝 上下文（Context）", placeholder="请输入参考内容……", lines=6)
        question_input = gr.Textbox(label="❓ 问题（Question）", placeholder="请输入你的问题……", lines=2)

    answer_output = gr.Textbox(label="📌 答案", lines=2)

    submit_btn = gr.Button("提交")

    submit_btn.click(fn=answer_question, inputs=[context_input, question_input], outputs=answer_output)

# 启动应用
demo.launch()


