import gradio as gr
from PIL import Image
import torch
from preprocessing import get, extract_features, gram, criterion

configue = {
    "lr": 0.003,
    "epochs": 2000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "image_shape": (512, 768),
    "epoch_step": 999999,
    "steps": 400,
}


def style_transfer_with_progress(content_img: Image.Image, style_img: Image.Image):
    content_X, style_X, layers, net = get(content_img, style_img, configue["device"], configue["image_shape"])

    with torch.no_grad():
        _, content_features = extract_features(net, content_X, layers)
        style_features, _ = extract_features(net, style_X, layers)
        style_gram = [gram(x) for x in style_features]

    X = content_X.clone().detach().requires_grad_(True)
    optim = torch.optim.AdamW([X], lr=configue["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optim, gamma=0.8, step_size=configue["steps"])

    total_steps = configue["epochs"]

    # 实时生成器
    for step in range(total_steps):
        optim.zero_grad()
        style_pred, content_pred = extract_features(net, X, layers)
        contents_l, style_l, tv_l, sum_l = criterion(X, content_features, content_pred, style_gram, style_pred)
        sum_l.backward()
        optim.step()
        scheduler.step()

        # 每 50 步更新进度条
        if step % 50 == 0 or step == total_steps - 1:
            current = X.clone().squeeze(0).cpu()
            current = torch.clamp(current, 0, 1)
            result_pil = Image.fromarray((current.permute(1, 2, 0).detach().numpy() * 255).astype('uint8'))

            progress_text = f"风格融合中... {step + 1}/{total_steps} 步"
            if step % 200 == 0 and step > 0:
                progress_text += " · 已更新中间结果！"

            yield result_pil, progress_text
    print("transformed...")

# ============== 用 Blocks 实现实时输出 ==============
with gr.Blocks(title="AI风格迁移 · 实时进度") as demo:
    gr.Markdown("# AI艺术风格迁移神器\n拖两张图，看魔法实时发生！")

    with gr.Row():
        content_input = gr.Image(label="内容图", type="pil")
        style_input = gr.Image(label="风格图", type="pil")

    submit_btn = gr.Button("开始生成艺术作品", variant="primary")
    output_image = gr.Image(label="实时结果（每50步更新一次）")
    status_text = gr.Textbox(label="进度", interactive=False)

    submit_btn.click(
        fn=style_transfer_with_progress,
        inputs=[content_input, style_input],
        outputs=[output_image, status_text]
    )

demo.queue()
demo.launch(server_name="0.0.0.0", share=True)