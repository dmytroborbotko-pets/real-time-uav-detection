# UAV Tracker: browser demo

Runs the fine-tuned YOLO11s detector fully client-side in the browser (ONNX Runtime Web, WebGPU
with a WASM fallback). Nothing is inferred server-side. There are two modes:

- **Тестове відео** plays `sample/infrared_test.mp4` (the same clip used in Day 3/4,
  `Anti-UAV-RGBT/test/20190926_134054_1_1/infrared.mp4`, re-encoded to H.264, see the note
  below) directly in the page and tracks the drone in it. You don't need a camera for this.
- **Камера** uses the device's rear camera (`facingMode: "environment"`) live. Point it at a
  second screen playing the IR test video rather than at the real sky: the model only ever saw
  infrared frames from a specialized long-range thermal camera, never daylight RGB.

## Run it

```bash
cd web
python3 serve.py 8000
```

`serve.py` is a tiny wrapper around Python's stdlib `http.server` that adds
`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers, which lets ONNX Runtime
Web use threaded and SIMD WASM (faster) when WebGPU isn't available. Plain
`python3 -m http.server` also works, just slower on the WASM fallback path.

Open `http://localhost:8000/index.html`. File mode works immediately on the same machine
(verified at about 24 FPS on an M1 Pro through the WebGPU backend, with the box tracking the
drone across frames), and needs no webcam.

### Getting it onto the iPhone (camera mode)

`getUserMedia` requires a secure context, and the iPhone is a different device than the dev
machine, so `http://localhost` doesn't count. You need a real HTTPS URL.

```bash
brew install cloudflared   # one-time
cloudflared tunnel --url http://localhost:8000
```

This prints a `https://*.trycloudflare.com` URL. Open that directly in iPhone Safari, tap
**Камера**, tap **Старт**, and grant camera access.

> If `cloudflared` fails with `failed to request quick Tunnel: ... context deadline exceeded`:
> Cloudflare's free quick-tunnel API (`api.trycloudflare.com`) comes with no uptime guarantee
> and can be flaky or blocked on some networks and DNS filters. The fallback, which also needs
> no signup:
> ```bash
> npx localtunnel --port 8000
> ```
> It prints a `https://*.loca.lt` URL. The first visit from the phone shows an interstitial
> "friendly reminder" page, so tap **Continue**. If it asks for a tunnel password, run
> `curl https://loca.lt/mytunnelpassword` on the machine running the tunnel and paste the result
> in.

### Filming tips (camera mode)

- Raise the second screen's brightness and dim the room. This reduces moiré and glare from
  re-photographing a display.
- Hold the phone steady, reasonably close to the screen and parallel to it.
- You can't switch mode after pressing Start, because stream teardown isn't implemented. Reload
  the page to switch between file and camera mode.

## Codec gotcha (already fixed, documented for next time)

The raw Anti-UAV `infrared.mp4` files are encoded as `FMP4` (MPEG-4 Part 2, old-style,
`fourcc=FMP4`). Chrome's `<video>` element refuses to decode this
(`DEMUXER_ERROR_NO_SUPPORTED_STREAMS`) even though `ffprobe` and OpenCV read it fine.
`sample/infrared_test.mp4` here is a re-encode to H.264/yuv420p:

```bash
ffmpeg -y -i ../Anti-UAV-RGBT/test/20190926_134054_1_1/infrared.mp4 \
  -c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p -movflags +faststart -an \
  sample/infrared_test.mp4
```

If you swap in a different test clip, re-encode it the same way first.

## How it works

- `notebooks/06_onnx_export.ipynb` exports `best.pt` → `model/best.onnx`. NMS is baked into the
  graph, which Ultralytics enables by default for detection models, so `app.js` only does
  box-coordinate unscaling and needs no NMS or IoU code of its own.
- `app.js` letterbox-resizes each video frame to 640×640 (matching the training `imgsz`), runs
  it through `onnxruntime-web`, unscales the returned boxes back to the video's native size, and
  draws them on an overlay `<canvas>`. The backend picks WebGPU if available and falls back to
  WASM. The FPS shown live on screen is the actual on-device number rather than an estimate.
- If FPS reads too low to feel live on the phone, try dropping `INPUT_SIZE` in `app.js` from 640
  to 416 or 320. That trades away some small-object accuracy, which is the exact weak spot the
  Day 4 failure analysis flagged. Also worth checking whether WebGPU is actually active (see the
  `backend:` label in the UI) or whether it silently fell back to WASM.