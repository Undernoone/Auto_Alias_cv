from __future__ import annotations

import argparse
import cgi
import html
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_MODEL_PATH = (
    r"C:\Users\WOUKEE\.cache\huggingface\hub\models--Qwen--Qwen2.5-VL-7B-Instruct"
    r"\snapshots\cc594898137f460bfe9f0759e9844b3ce807cfb5"
)

_MODEL: Any | None = None
_PROCESSOR: Any | None = None
_DEVICE = "cuda:0"
_MODEL_PATH = DEFAULT_MODEL_PATH
_MODEL_LOCK = threading.Lock()


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Qwen2.5-VL web test without Gradio.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--upload-dir", default=r"F:\430AutoAlias\vlm_web_uploads")
    args = parser.parse_args()

    global _DEVICE, _MODEL_PATH
    _DEVICE = args.device
    _MODEL_PATH = args.model

    upload_dir = Path(args.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    handler = _make_handler(upload_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Qwen2.5-VL local web: http://{args.host}:{args.port}/", flush=True)
    print(f"Model: {_MODEL_PATH}", flush=True)
    print(f"Uploads: {upload_dir}", flush=True)
    server.serve_forever()
    return 0


def _make_handler(upload_dir: Path):
    class QwenLocalHandler(BaseHTTPRequestHandler):
        server_version = "QwenLocalWeb/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(_page())
                return
            if parsed.path == "/health":
                self._send_json({"ok": True, "model_loaded": _MODEL is not None})
                return
            self.send_error(404, "not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/ask":
                self.send_error(404, "not found")
                return
            started = time.time()
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    },
                )
                prompt = _field_text(form, "prompt") or "Describe this image."
                max_new_tokens = int(_field_text(form, "max_new_tokens") or "512")
                image_item = form["image"] if "image" in form else None
                if image_item is None or not getattr(image_item, "filename", ""):
                    raise ValueError("Please upload an image.")
                suffix = Path(image_item.filename).suffix.lower() or ".png"
                image_path = upload_dir / f"qwen_input_{int(time.time() * 1000)}{suffix}"
                data = image_item.file.read()
                if not data:
                    raise ValueError("Uploaded image is empty.")
                image_path.write_bytes(data)
                answer = _ask_qwen(image_path, prompt, max_new_tokens=max_new_tokens)
                elapsed = time.time() - started
                self._send_html(_page(prompt=prompt, answer=answer, image_path=str(image_path), elapsed=elapsed))
            except Exception as exc:
                elapsed = time.time() - started
                self._send_html(_page(error=str(exc), elapsed=elapsed), status=500)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_html(self, body: str, *, status: int = 200) -> None:
            encoded = body.encode("utf-8", errors="replace")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return QwenLocalHandler


def _field_text(form: cgi.FieldStorage, name: str) -> str:
    if name not in form:
        return ""
    item = form[name]
    if isinstance(item, list):
        item = item[0]
    value = getattr(item, "value", "")
    return str(value)


def _load_model() -> tuple[Any, Any]:
    global _MODEL, _PROCESSOR
    with _MODEL_LOCK:
        if _MODEL is not None and _PROCESSOR is not None:
            return _MODEL, _PROCESSOR
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "local_files_only": True,
        }
        if _DEVICE.startswith("cuda") and torch.cuda.is_available():
            model_kwargs["device_map"] = {"": _DEVICE}
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(_MODEL_PATH, **model_kwargs)
        if "device_map" not in model_kwargs:
            model = model.to(_DEVICE if _DEVICE != "auto" else "cpu")
        model.eval()
        processor = AutoProcessor.from_pretrained(_MODEL_PATH, local_files_only=True)
        _MODEL = model
        _PROCESSOR = processor
        return model, processor


def _ask_qwen(image_path: Path, prompt: str, *, max_new_tokens: int) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    model, processor = _load_model()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    target_device = getattr(model, "device", None)
    if target_device is not None:
        inputs = inputs.to(target_device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max(32, min(int(max_new_tokens), 4096)),
            do_sample=False,
        )
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated, strict=False)
    ]
    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip() if decoded else ""


def _page(
    *,
    prompt: str = "Describe this image and list visible automotive design curves.",
    answer: str = "",
    image_path: str = "",
    error: str = "",
    elapsed: float | None = None,
) -> str:
    escaped_prompt = html.escape(prompt)
    escaped_answer = html.escape(answer)
    escaped_error = html.escape(error)
    escaped_image = html.escape(image_path)
    elapsed_text = f"{elapsed:.1f}s" if elapsed is not None else ""
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Qwen2.5-VL Local Test</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; background: #f6f7f9; color: #172033; }}
    .wrap {{ max-width: 980px; margin: 0 auto; background: #fff; padding: 22px; border: 1px solid #d9dde6; }}
    textarea {{ width: 100%; height: 130px; font-size: 14px; }}
    input, button {{ font-size: 14px; }}
    button {{ padding: 9px 16px; margin-top: 12px; }}
    pre {{ white-space: pre-wrap; background: #101828; color: #e6edf7; padding: 16px; overflow:auto; }}
    .err {{ background: #fff1f1; color: #9b1c1c; padding: 12px; border: 1px solid #f0b5b5; }}
    .hint {{ color: #667085; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Qwen2.5-VL Local Test</h2>
    <p class="hint">Model: {html.escape(_MODEL_PATH)} | Device: {html.escape(_DEVICE)}</p>
    <form action="/ask" method="post" enctype="multipart/form-data">
      <p><input type="file" name="image" accept="image/*" required></p>
      <p><textarea name="prompt">{escaped_prompt}</textarea></p>
      <p>max_new_tokens <input type="number" name="max_new_tokens" value="512" min="32" max="4096"></p>
      <button type="submit">Ask Qwen2.5-VL</button>
    </form>
    <p class="hint">First request loads the model, so it can take a while. Elapsed: {elapsed_text}</p>
    {f'<div class="err">{escaped_error}</div>' if error else ''}
    {f'<p class="hint">Uploaded: {escaped_image}</p>' if image_path else ''}
    {f'<h3>Answer</h3><pre>{escaped_answer}</pre>' if answer else ''}
  </div>
</body>
</html>"""


if __name__ == "__main__":
    raise SystemExit(main())
