import base64
import io
import os

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, ImageDraw

app = Flask(__name__)

def perspective_coeffs(src_quad, dst_quad):
    matrix = []
    for (x, y), (X, Y) in zip(dst_quad, src_quad):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(src_quad, dtype=np.float64).reshape(8)
    res = np.linalg.solve(A, B)
    return res.tolist()

def warp_design(design_img, base_size, quad):
    w, h = design_img.size
    src_quad = [(0, 0), (w, 0), (w, h), (0, h)]
    coeffs = perspective_coeffs(src_quad, quad)
    return design_img.transform(
        base_size,
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )

def build_shadow_map(base_img, quad):
    gray = base_img.convert("L")
    arr = np.array(gray, dtype=np.float64)
    mask_img = Image.new("L", base_img.size, 0)
    ImageDraw.Draw(mask_img).polygon(quad, fill=255)
    mask = np.array(mask_img) > 0
    region_vals = arr[mask]
    if region_vals.size == 0:
        return Image.new("L", base_img.size, 128)
    lo, hi = np.percentile(region_vals, [2, 98])
    if hi - lo < 1:
        normalized = np.full_like(arr, 128.0)
    else:
        normalized = (arr - lo) / (hi - lo) * 255.0
        normalized = np.clip(normalized, 0, 255)
    return Image.fromarray(normalized.astype(np.uint8), mode="L")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/warp", methods=["POST"])
def warp():
    data = request.get_json(force=True)
    required = ["blank_b64", "design_b64", "quad", "blend_mode", "opacity"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400
    try:
        blank_bytes = base64.b64decode(data["blank_b64"])
        design_bytes = base64.b64decode(data["design_b64"])
        quad = [tuple(pt) for pt in data["quad"]]
        blend_mode = data["blend_mode"]
        opacity = float(data["opacity"])
        base = Image.open(io.BytesIO(blank_bytes)).convert("RGBA")
        design = Image.open(io.BytesIO(design_bytes)).convert("RGBA")
        warped = warp_design(design, base.size, quad)
        if blend_mode == "direct":
            result = base.copy()
            result.alpha_composite(warped)
        elif blend_mode == "multiply_shadow":
            shadow_map = build_shadow_map(base, quad)
            warped_rgb = np.array(warped.convert("RGB"), dtype=np.float64)
            shadow_arr = np.array(shadow_map, dtype=np.float64)[:, :, None] / 255.0
            lit = warped_rgb * (shadow_arr * 1.15)
            lit = np.clip(lit, 0, 255).astype(np.uint8)
            lit_img = Image.fromarray(lit, mode="RGB").convert("RGBA")
            alpha = warped.split()[-1].point(lambda a: int(a * opacity))
            lit_img.putalpha(alpha)
            result = base.copy()
            result.alpha_composite(lit_img)
        else:
            return jsonify({"error": f"Unknown blend_mode: {blend_mode}"}), 400
        out = io.BytesIO()
        result.convert("RGB").save(out, format="PNG")
        out.seek(0)
        result_b64 = base64.b64encode(out.read()).decode("utf-8")
        return jsonify({"composited_b64": result_b64, "width": result.width, "height": result.height})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Warp service starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
