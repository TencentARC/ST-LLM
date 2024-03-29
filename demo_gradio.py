import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

import argparse
import torch

from stllm.common.config import Config
from stllm.common.registry import registry
from stllm.conversation.conversation import Chat, CONV_instructblip_Vicuna0

# imports modules for registration
from stllm.datasets.builders import *
from stllm.models import *
from stllm.processors import *
from stllm.runners import *
from stllm.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='config/instructblipbase_stllm_conversation.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--ckpt-path", required=True, help="path to STLLM_conversation_weight.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

ckpt_path = args.ckpt_path
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_config.ckpt = ckpt_path
model_config.llama_model = ckpt_path
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.to(torch.float16)
CONV_VISION = CONV_instructblip_Vicuna0

chat = Chat(model, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list


def upload_video(gr_video, chat_state, num_segments, text_prompt='Watch the video and answer the question.'):
    print('gr_video: ', gr_video)
    img_list = []
    if gr_video: 
        chat_state = CONV_VISION.copy()
        chat.upload_video(gr_video, chat_state, img_list, num_segments, text=text_prompt)
        return gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state, gr_video, num_segments):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_video(gr_video, chat_state, img_list, num_segments, text=user_message)
    msg = "###Human: " + user_message + " ###Assistant: "
    chat.ask(msg, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state, img_list


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, do_sample=False, temperature=temperature, max_length=2000)[0]
    llm_message = llm_message.replace("<s>", "") # handle <s>
    chatbot[-1][1] = llm_message
    print(chat_state)
    print(f"Answer: {llm_message}")
    return chatbot, chat_state, img_list


class STLLM(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )


gvlabtheme = STLLM(primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        )

title = """<h1 align="center"><a href="https://github.com/farewellthree/ST-LLM"><img src="https://s21.ax1x.com/2024/03/25/pF4Wzq0.png" border="0" style="margin: 0 auto; height: 150px;" /></a> </h1>"""
description ="""
        CLICK FOR SOURCE CODE!<br><p><a href='https://github.com/farewellthree/ST-LLM'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p>
        """


with gr.Blocks(title="ST-LLM Chatbot!",theme=gvlabtheme,css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5, visible=True) as video_upload:
            with gr.Column(elem_id="image", scale=0.5) as img_part:
                with gr.Tab("Video", elem_id='video_tab'):
                    up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload").style(height=360)
            # text_prompt_input = gr.Textbox(value="Watch the video and answer the question.",show_label=False, placeholder='Input your text prompt, example: "Watch the video and answer the question."', interactive=True).style(container=False)           
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                interactive=True,
                label="beam search numbers",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
            
            num_segments = gr.Slider(
                minimum=16,
                maximum=96,
                value=64,
                step=1,
                interactive=True,
                label="Video Segments",
            )
        
        with gr.Column(visible=True)  as input_raws:
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(elem_id="chatbot",label='ST-LLM')
            with gr.Row():
                with gr.Column(scale=0.7):
                    text_input = gr.Textbox(show_label=False, placeholder='Please upload your video first', interactive=False).style(container=False)
                with gr.Column(scale=0.15, min_width=0):
                    run = gr.Button("💭Send")
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("🔄Clear️")     
    
    upload_button.click(upload_video, [up_video, chat_state, num_segments], [up_video, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state, up_video, num_segments], [text_input, chatbot, chat_state, img_list]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    run.click(gradio_ask, [text_input, chatbot, chat_state, up_video, num_segments], [text_input, chatbot, chat_state, img_list]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    run.click(lambda: "", None, text_input)  
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, up_video, text_input, upload_button, chat_state, img_list], queue=False)

demo.launch(share=True, enable_queue=True)
