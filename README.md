
# StableLive // Hybrid Engine

**A High-Performance Real-Time VJ Tool using SDXL Turbo.**

* **Architecture:** Hybrid Python (Logic) + Rust (Image Processing).
* **Protocol:** Raw Binary over WebSockets (Zero-Copy).
* **Engine:** StabilityAI SDXL Turbo + TensorRT/CUDA.

---

## 📂 Project Structure

Ensure your folder looks exactly like this before starting:

```text
StableLive/
├── hyper_stable/           <-- [Folder] Your Rust extension
│   ├── Cargo.toml          <-- Rust config
│   └── src/
│       └── lib.rs          <-- Rust source code
├── server.py               <-- Main Python Server
├── index.html              <-- The Browser Interface
└── venv/                   <-- [Folder] Python Virtual Environment

```

---

## 🛠️ Phase 1: Prerequisites

You need these installed on your system before anything will work.

**1. Install Rust (The Compiler)**
The engine needs this to build `hyper_stable`.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Restart your terminal after installing!

```

**2. Install Python 3.10+**
Ensure you have Python installed.

```bash
python3 --version

```

**3. NVIDIA Drivers & CUDA**
Ensure your RTX 3090 drivers are up to date.

---

## ⚙️ Phase 2: Installation

**1. Create & Activate Virtual Environment**

```bash
# In the StableLive folder
python3 -m venv venv
source venv/bin/activate

```

**2. Install Python Dependencies**
Install the AI libraries and the Rust bridge builder.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate opencv-python numpy uvicorn fastapi websockets maturin

```

*(Note: If you are on CUDA 12, the first line will automatically find the right version, or check PyTorch.org)*

**3. Compile the Rust Engine**
This builds your custom `hyper_stable` module and installs it into your venv.

```bash
cd hyper_stable
maturin develop --release
cd ..

```

* **Crucial:** Do not forget `--release`. If you skip it, the engine will be in "Debug Mode" and run 100x slower.
* *Expected Output:* `📦 Built wheel for hyper_stable... Installed package hyper_stable`.

---

## 🚀 Phase 3: Running the Engine

**1. Start the Server**
Ensure you are in the main folder (where `server.py` is) and your venv is active.

```bash
python server.py

```

**2. Open the Client**
Open your web browser (Chrome/Edge recommended) and go to:
`http://localhost:8000`

---

## 🌡️ Troubleshooting & Notes

**"The CPU is overheating!"**

* **Cause:** `torch.compile` uses AVX instructions that generate massive heat.
* **Fix:** Ensure `pipe.unet = torch.compile(...)` is commented out in `server.py` until your new Noctua cooler arrives.
* **Status:** The current script uses "Eco Mode" settings (384p Preview) to be safe.

**"I see 'ModuleNotFoundError: No module named hyper_stable'"**

* **Fix:** You forgot to compile the Rust module or your venv wasn't active when you ran `maturin`.
* **Action:**
1. `source venv/bin/activate`
2. `cd hyper_stable`
3. `maturin develop --release`



**"FPS is low (<15)"**

* **Check:** Look at the terminal output.
* **If GPU time is high:** Enable `torch.compile` (Requires good cooling).
* **If Network Lag:** The system is already using Binary Mode. Ensure you aren't on Wi-Fi if testing remotely (use Ethernet or Localhost).

---
