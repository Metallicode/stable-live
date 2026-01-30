import asyncio
import torch
import time
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderTiny
from torchvision.transforms import functional as TF

# --- YOUR CUSTOM ENGINE ---
import hyper_stable 
# --------------------------

# --- CONFIG ---
TARGET_FPS = 30       
JPEG_QUALITY = 75     
PREVIEW_SIZE = 384    # We let Rust handle the resize to this target
# --------------

app = FastAPI()

print("Loading SDXL Turbo (HYPER STABLE ENGINE)...")

# 1. Load SDXL Turbo
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
pipe.to("cuda")

# SAFETY: Keeping compile off until your cooler arrives
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

print("Ready. Engine: RUST.")

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
    if rotate != 0:
        tensor = TF.rotate(tensor, rotate, interpolation=TF.InterpolationMode.NEAREST)
    if zoom != 1.0 or pan_x != 0 or pan_y != 0:
        tensor = TF.affine(tensor, angle=0, translate=[pan_x, pan_y], scale=zoom, shear=0, interpolation=TF.InterpolationMode.NEAREST)
    return tensor

def get_mixed_embeddings(pipe, prompt_a, prompt_b, ratio):
    with torch.no_grad():
        (e_a, n_e_a, p_a, n_p_a) = pipe.encode_prompt(prompt=prompt_a, negative_prompt=current_state["neg_prompt"])
        (e_b, _, p_b, _) = pipe.encode_prompt(prompt=prompt_b, negative_prompt=current_state["neg_prompt"])
        return (e_a * (1 - ratio)) + (e_b * ratio), (p_a * (1 - ratio)) + (p_b * ratio), n_e_a, n_p_a

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
            except Exception:
                break

            # 1. Transform & Generate (GPU)
            input_tensor = apply_gpu_transform(
                last_generated_tensor, 
                current_state["zoom"], current_state["rotate"], 
                current_state["pan_x"], current_state["pan_y"]
            )
            
            p, pp, n, np = get_mixed_embeddings(pipe, current_state["prompt_a"], current_state["prompt_b"], current_state["mix_ratio"])
            
            try:
                with torch.inference_mode():
                    results = pipe(
                        prompt_embeds=p, pooled_prompt_embeds=pp, negative_prompt_embeds=n, negative_pooled_prompt_embeds=np,
                        image=input_tensor, 
                        strength=current_state["strength"], 
                        guidance_scale=0.0, 
                        num_inference_steps=2,
                        output_type="pt" 
                    )
                if len(results.images) > 0:
                    last_generated_tensor = results.images 
            except Exception:
                pass 

            # 2. PREPARE FOR RUST (The Bridge)
            # Rust needs: uint8, CPU, Shape (Height, Width, 3)
            # We do the bare minimum here in Python to get the data ready
            
            # A. Normalize to 0-1 and Cast to Byte (Still on GPU)
            # Note: We keep it 512x512 here and let Rust downscale
            tensor_u8 = (last_generated_tensor.squeeze(0) / 2 + 0.5).clamp(0, 1).mul(255).byte()
            
            # B. Move to CPU and Permute to HWC (Channels Last)
            # This is the "Data Transfer" cost, unavoidable.
            array_np = tensor_u8.permute(1, 2, 0).cpu().numpy()

            # 3. RUST ENGINE EXECUTION
            # This runs in C++, releases the GIL, and is extremely fast.
            # process_frame(array, target_w, target_h, quality)
            jpeg_bytes = hyper_stable.process_frame(
                array_np, 
                PREVIEW_SIZE, 
                PREVIEW_SIZE, 
                JPEG_QUALITY
            )
            
            # 4. Send Bytes
            await websocket.send_bytes(jpeg_bytes)
            
            # 5. FPS Lock
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / TARGET_FPS) - elapsed
            if sleep_time > 0: await asyncio.sleep(sleep_time)
            else: await asyncio.sleep(0)

    except Exception as e:
        print(f"Disconnected: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
