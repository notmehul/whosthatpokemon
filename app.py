import gradio as gr
from fastai.vision.all import *
from PIL import Image


learn = load_learner('model.pkl')
labels = learn.dls.vocab

def whoisit(image):
    image=PILImage.create(image)
    pred,pred_idx,probs = learn.predict(image)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Who's that pokemon"
description = "pokemon classifier bruh."
examples = ['f403dcd3136643e4b43c7d01e14494bc.jpg', '5e623006ed0e4a23b72441aa8cf3e52c.jpg', '00000034.jpg', '1d18c0d07974465091b45383db76dda1.jpg']
interpretation='default'
enable_queue=True


gr.Interface(fn=whoisit,
             inputs=gr.Image(type="pil",height=512, width=512),
             outputs=gr.Label(num_top_classes=3),
             examples=examples).launch(share=True)