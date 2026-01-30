import asyncio
import torch
import time
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderTiny
from torchvision.transforms import functional as TF
import io
import base64
import uvicorn

# --- TUNING CONFIG ---
TARGET_FPS = 24       # Cinematic framerate (Smooth but not overheating)
JPEG_QUALITY = 70     # Good balance
PREVIEW_SIZE = 384    # We Generate at 512, but send 384 to save CPU heat
# ---------------------

app = FastAPI()

print("Loading SDXL Turbo (Quality + Cooling Hybrid)...")

# 1. Load SDXL Turbo
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
)

# 2. Load TinyVAE (Critical for SDXL Speed)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)

pipe.to("cuda")

print(f"Ready. Target: {TARGET_FPS} FPS. CPU Safeguards Active.")

# Initialize GPU Tensor (Batch, Channels, Height, Width)
last_generated_tensor = torch.zeros((1, 3, 512, 512), device="cuda", dtype=torch.float16)

current_state = {
    "prompt_a": "cyberpunk city, neon lights", 
    "prompt_b": "ancient jungle ruins, stone", 
    "mix_ratio": 0.0, 
    "neg_prompt": "text, watermark, blur, low quality",
    "strength": 0.5, 
    "zoom": 1.0, "rotate": 0.0, "pan_x": 0, "pan_y": 0,
    "resolution": 512 
}

# --- GPU TRANSFORMS ---
def apply_gpu_transform(tensor, zoom, rotate, pan_x, pan_y):
    # This runs entirely on the 3090. Zero CPU usage.
    if rotate != 0:
        tensor = TF.rotate(tensor, rotate, interpolation=TF.InterpolationMode.NEAREST)
    
    if zoom != 1.0 or pan_x != 0 or pan_y != 0:
        tensor = TF.affine(
            tensor, 
            angle=0, 
            translate=[pan_x, pan_y], 
            scale=zoom, 
            shear=0,
            interpolation=TF.InterpolationMode.NEAREST
        )
    return tensor

def get_mixed_embeddings(pipe, prompt_a, prompt_b, ratio):
    with torch.no_grad():
        (e_a, n_e_a, p_a, n_p_a) = pipe.encode_prompt(prompt=prompt_a, negative_prompt=current_state["neg_prompt"])
        (e_b, _, p_b, _) = pipe.encode_prompt(prompt=prompt_b, negative_prompt=current_state["neg_prompt"])
        return (e_a * (1 - ratio)) + (e_b * ratio), (p_a * (1 - ratio)) + (p_b * ratio), n_e_a, n_p_a

# --- OPTIMIZED ENCODING ---
def encode_tensor_to_base64(tensor):
    # 1. Resize on GPU first! (Saving CPU work)
    # We downscale from 512 to 384 using the GPU's power
    if tensor.shape[-1] != PREVIEW_SIZE:
        tensor = torch.nn.functional.interpolate(tensor, size=PREVIEW_SIZE, mode='nearest')

    # 2. Move to CPU & Normalize
    arr = (tensor.squeeze(0) / 2 + 0.5).clamp(0, 1).cpu().float().numpy()
    
    # 3. Shape to HWC
    arr = np.transpose(arr, (1, 2, 0)) * 255
    
    # 4. RGB -> BGR (OpenCV)
    arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # 5. JPEG Encode (OpenCV C++ backend)
    _, buffer = cv2.imencode('.jpg', arr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return base64.b64encode(buffer).decode("utf-8")

@app.get("/")
async def get():
    with open("index.html", "r") as f: return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global last_generated_tensor
    
    try:
        while True:
            loop_start = time.time()

            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.001)
                current_state.update(data)
            except asyncio.TimeoutError:
                pass 

            res = current_state["resolution"]

            # 1. Transform (GPU)
            input_tensor = apply_gpu_transform(
                last_generated_tensor, 
                current_state["zoom"], current_state["rotate"], 
                current_state["pan_x"], current_state["pan_y"]
            )
            
            # 2. Embeddings (GPU)
            p, pp, n, np = get_mixed_embeddings(pipe, current_state["prompt_a"], current_state["prompt_b"], current_state["mix_ratio"])
            
            # 3. Generate (SDXL Turbo)
            try:
                with torch.inference_mode():
                    results = pipe(
                        prompt_embeds=p, pooled_prompt_embeds=pp, negative_prompt_embeds=n, negative_pooled_prompt_embeds=np,
                        image=input_tensor, 
                        strength=current_state["strength"], 
                        guidance_scale=0.0, 
                        num_inference_steps=10, # Safe buffer
                        output_type="pt"        # Keep on GPU
                    )
                
                if len(results.images) > 0:
                    last_generated_tensor = results.images 

            except Exception as e:
                pass # Ignore occasional math errors

            # 4. Send (OpenCV + GPU Resize)
            await websocket.send_text(encode_tensor_to_base64(last_generated_tensor))
            
            # 5. Smart Sleep (Target FPS)
            # This keeps your CPU temp stable by forcing it to rest
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / TARGET_FPS) - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0)

    except Exception as e:
        print(f"Disconnected: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
