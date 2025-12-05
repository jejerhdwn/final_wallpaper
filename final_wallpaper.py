import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import colorsys
import random


# =========================
# 팔레트 추출 (k-means 비슷한 방식)
# =========================

def extract_palette_from_images(images, num_colors: int = 5, max_samples: int = 8000):
    """
    여러(또는 한) 이미지에서 공통 팔레트 추출.
    '많이 나온 픽셀색'만 쓰지 않고, 색 덩어리를 k-means처럼 묶어서 중심색을 뽑음.
    """
    all_pixels = []

    for image in images:
        img = image.convert("RGB")
        img = img.resize((220, 220))
        arr = np.array(img, dtype=np.float32) / 255.0
        pixels = arr.reshape(-1, 3)
        all_pixels.append(pixels)

    if not all_pixels:
        return np.array([])

    pixels = np.vstack(all_pixels)
    n_pixels = pixels.shape[0]

    if n_pixels > max_samples:
        idx = np.random.choice(n_pixels, max_samples, replace=False)
        pixels = pixels[idx]

    k = min(num_colors, len(pixels))
    rng = np.random.default_rng(42)
    centers = pixels[rng.choice(len(pixels), k, replace=False)]

    for _ in range(12):
        dists = np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        new_centers = []
        for j in range(k):
            cluster_pixels = pixels[labels == j]
            if len(cluster_pixels) == 0:
                new_centers.append(centers[j])
            else:
                new_centers.append(cluster_pixels.mean(axis=0))
        new_centers = np.stack(new_centers, axis=0)

        if np.allclose(new_centers, centers, atol=1e-3):
            centers = new_centers
            break
        centers = new_centers

    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    centers = centers[order]

    centers = np.clip(centers, 0.0, 1.0)
    return centers


def plot_palette(colors):
    """색상 팔레트를 matplotlib으로 시각화"""
    if colors.size == 0:
        return None
    num_colors = len(colors)
    fig, ax = plt.subplots(figsize=(num_colors * 1.2, 1.5))
    ax.set_xlim(0, num_colors)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, rgb in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=rgb))
    return fig


def colors_to_hex_list(colors):
    hex_list = []
    for rgb in colors:
        r, g, b = (rgb * 255).astype(int)
        hex_list.append(f"#{r:02X}{g:02X}{b:02X}")
    return hex_list


# =========================
# 컬러 / 무드 관련
# =========================

def adjust_colors_with_mood(colors, brightness_level, saturation_level):
    """
    0~1 brightness / saturation 슬라이더 값으로 팔레트 전체 톤 조정
    """
    if colors.size == 0:
        return colors

    adjusted = []
    for rgb in colors:
        r, g, b = rgb
        h, l, s = colorsys.rgb_to_hls(r, g, b)

        l = (l * 0.5) + (brightness_level * 0.5)
        s = (s * 0.4) + (saturation_level * 0.6)

        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        adjusted.append([r2, g2, b2])

    return np.clip(np.array(adjusted), 0.0, 1.0)


def describe_mood_params(brightness, saturation, abstractness):
    def level_desc(x):
        if x < 0.33:
            return "낮음"
        elif x < 0.66:
            return "중간"
        else:
            return "높음"

    return (
        f"밝기: {level_desc(brightness)}, "
        f"채도: {level_desc(saturation)}, "
        f"추상 정도: {level_desc(abstractness)}"
    )


def heuristic_mood_description(colors, brightness, saturation, abstractness):
    """팔레트 + 슬라이더 값으로 간단한 무드 설명"""
    if colors.size == 0:
        return "이미지에서 색상을 충분히 추출하지 못했습니다. 기본 중립 톤으로 배경을 생성합니다."

    rgbs = colors
    hs = []
    luminances = []
    for rgb in rgbs:
        r, g, b = rgb
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        hs.append(h)
        luminances.append(l)

    avg_h = float(np.mean(hs))
    avg_l = float(np.mean(luminances))

    warmth = "중립적인"
    if (avg_h < 0.13) or (avg_h > 0.8):
        warmth = "따뜻한"
    elif 0.3 < avg_h < 0.7:
        warmth = "차가운"

    if avg_l > 0.7:
        tone_desc = "밝고 가벼운 톤"
    elif avg_l < 0.4:
        tone_desc = "어둡고 무게감 있는 톤"
    else:
        tone_desc = "중간 톤"

    if abstractness < 0.33:
        abs_desc = "현실적인 분위기에 가깝게"
    elif abstractness < 0.66:
        abs_desc = "약간 추상적인 느낌으로"
    else:
        abs_desc = "형태보다 색과 리듬이 강조되는 추상적인 느낌으로"

    hex_colors = colors_to_hex_list(colors)
    lines = [
        f"- 전체적으로 {warmth} 무드와 {tone_desc}입니다.",
        f"- 대표 색상(대략적인 팔레트): {', '.join(hex_colors[:5])}",
        f"- 설정한 무드 파라미터를 반영해 {abs_desc} 배경이 만들어집니다.",
    ]
    return "\n".join(lines)


# =========================
# 공용: 비율 맞춰 자르기
# =========================

def crop_to_aspect(img, target_size):
    """원본 이미지를 월페이퍼 비율에 맞게 중앙 크롭"""
    target_w, target_h = target_size
    target_ratio = target_w / target_h

    img = img.convert("RGB")
    w, h = img.size
    ratio = w / h

    if ratio > target_ratio:
        new_w = int(h * target_ratio)
        new_h = h
    else:
        new_w = w
        new_h = int(w / target_ratio)

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    img_cropped = img.crop((left, top, right, bottom))
    img_resized = img_cropped.resize(target_size, Image.LANCZOS)
    return img_resized


# =========================
# 1) 단색 배경
# =========================

def generate_solid_wallpaper(colors, size=(1024, 1792)):
    width, height = size
    img = Image.new("RGB", size)
    if colors.size == 0:
        color = (240, 240, 240)
    else:
        rgb = colors[0]
        color = tuple((rgb * 255).astype(int))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, height], fill=color)
    return img


# =========================
# 2) Soft: 원본 이미지를 강하게 블러한 배경
# =========================

def generate_soft_from_original(base_image, brightness_level, saturation_level, size=(1024, 1792)):
    """
    원본 사진을 월페이퍼 비율로 자른 뒤,
    알아볼 수 없을 정도로 강하게 블러 + 밝기/채도만 조절해서
    '무드만 남는 배경'으로 만들기
    """
    if base_image is None:
        return Image.new("RGB", size, (230, 230, 235))

    img = crop_to_aspect(base_image, size)

    # 디테일 완전 날리기
    img = img.filter(ImageFilter.GaussianBlur(radius=28))

    # 밝기/채도 보정
    b_factor = 0.7 + brightness_level * 0.7   # 0.7 ~ 1.4
    s_factor = 0.5 + saturation_level * 0.9   # 0.5 ~ 1.4

    img = ImageEnhance.Brightness(img).enhance(b_factor)
    img = ImageEnhance.Color(img).enhance(s_factor)

    return img


# =========================
# 3) Abstract: 수채화 / wobble 느낌 추상 배경
# =========================

def generate_abstract_background(colors, abstract_level, size=(1024, 1792)):
    """
    팔레트 색을 사용해서 수채화 느낌의 추상 배경 생성:
    - 팔레트의 두 색으로 세로 그라디언트 깔고
    - 반투명한 '물감 블롭'들을 여러 겹으로 얹은 뒤
    - 전체를 블러 + 살짝 그레인 추가
    """
    if colors.size == 0:
        colors = np.array([
            [0.82, 0.82, 0.88],
            [0.35, 0.40, 0.55],
            [0.93, 0.86, 0.80],
        ])

    width, height = size

    base1 = colors[0]
    base2 = colors[-1] if len(colors) > 1 else colors[0]

    h = height
    w = width
    grad = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        t = y / (h - 1)
        grad[y, :, :] = (1 - t) * base1 + t * base2

    grad_uint8 = (grad * 255).clip(0, 255).astype("uint8")
    img = Image.fromarray(grad_uint8, mode="RGB")

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    base_blobs = 25
    extra = int(abstract_level * 45)
    num_blobs = base_blobs + extra

    for i in range(num_blobs):
        rgb = colors[i % len(colors)]
        r, g, b = (rgb * 255).astype(int)

        alpha = random.randint(40, 110)
        color = (r, g, b, alpha)

        max_radius = int(min(w, h) * (0.25 + 0.25 * abstract_level))
        min_radius = int(min(w, h) * 0.08)
        rx = random.randint(min_radius, max_radius)
        ry = int(rx * random.uniform(0.6, 1.4))

        cx = random.randint(-int(w * 0.1), int(w * 1.1))
        cy = random.randint(-int(h * 0.1), int(h * 1.1))

        jitter_times = random.randint(2, 4)
        for _ in range(jitter_times):
            jx = int(cx + random.uniform(-rx * 0.15, rx * 0.15))
            jy = int(cy + random.uniform(-ry * 0.15, ry * 0.15))
            draw.ellipse([jx - rx, jy - ry, jx + rx, jy + ry], fill=color)

    composed = Image.alpha_composite(img.convert("RGBA"), overlay)

    blur_radius = 5 + abstract_level * 4
    composed = composed.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    arr = np.array(composed.convert("RGB")).astype("int16")
    noise_strength = 12
    noise = np.random.randint(-noise_strength, noise_strength + 1, size=arr.shape[:2] + (1,))
    arr = np.clip(arr + noise, 0, 255).astype("uint8")

    final_img = Image.fromarray(arr, mode="RGB")
    return final_img


# =========================
# 4) 패턴 템플릿: 업로드한 패턴 이미지를 톤만 맞춰서 사용
# =========================

def generate_pattern_from_template(pattern_img, colors, brightness_level, saturation_level, size=(1024, 1792)):
    """
    사용자가 업로드한 패턴 이미지를:
    - 월페이퍼 비율로 크롭/리사이즈
    - 밝기/채도 슬라이더 반영
    - 팔레트 대표 색으로 살짝 컬러 오버레이
    """
    if pattern_img is None:
        return Image.new("RGB", size, (230, 230, 235))

    img = crop_to_aspect(pattern_img, size)

    # 밝기/채도 조정
    b_factor = 0.7 + brightness_level * 0.7
    s_factor = 0.5 + saturation_level * 0.9
    img = ImageEnhance.Brightness(img).enhance(b_factor)
    img = ImageEnhance.Color(img).enhance(s_factor)

    # 팔레트 대표 색으로 아주 얇은 컬러 레이어
    if colors.size > 0:
        main = colors[0]
        r, g, b = (main * 255).astype(int)
        overlay = Image.new("RGBA", img.size, (r, g, b, 40))  # 투명한 레이어
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    return img


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Moodboard 월페이퍼 생성기 (템플릿 + Soft/Abstract)",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Moodboard 기반 월페이퍼 생성기")
st.write(
    """
이미지 **1장 또는 여러 장**을 업로드하면,  
공통된 **무드 & 컬러 팔레트**를 분석해서  
선택한 방식으로 **배경화면**을 생성합니다.

- 단색: 팔레트 대표 색으로 깔끔한 단색 배경  
- 비슷한 무드의 이미지 느낌 (Soft): 원본 사진을 많이 블러해서 '무드만 남는' 배경  
- 추상 (Abstract): 팔레트 색으로 만든 수채화 느낌 추상 배경  
- 패턴 템플릿: 직접 만든 패턴 이미지를 업로드해서, 무드에 맞게 톤만 조정
"""
)

generation_mode = st.sidebar.selectbox(
    "배경화면 타입 선택",
    [
        "단색 (Solid color)",
        "비슷한 무드의 이미지 느낌 (Soft)",
        "추상 배경화면 (Abstract)",
        "패턴 템플릿 (업로드 이미지 사용)",
    ],
)

num_palette_colors = st.sidebar.slider("팔레트 색상 개수", 3, 8, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("무드 파라미터")
brightness_level = st.sidebar.slider("Brightness (밝기)", 0.0, 1.0, 0.6, 0.05)
saturation_level = st.sidebar.slider("Saturation (채도)", 0.0, 1.0, 0.7, 0.05)
abstract_level = st.sidebar.slider("Abstractness (추상 정도)", 0.0, 1.0, 0.7, 0.05)

st.sidebar.markdown("---")
st.sidebar.write("1. 메인 이미지 업로드 → 2. (필요 시) 패턴 템플릿 업로드 → 3. 생성 버튼 클릭")

# 메인 이미지 업로더 (무드보드용)
uploaded_files = st.file_uploader(
    "무드를 만들 이미지를 업로드하세요 (1장 또는 여러 장, 룩북, OOTD, 레퍼런스 등)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

# 패턴 템플릿 업로더 (해당 모드일 때만 사용)
pattern_file = None
if "패턴 템플릿" in generation_mode:
    pattern_file = st.file_uploader(
        "패턴 템플릿 이미지를 업로드하세요 (체크/도트/텍스타일 등)",
        type=["png", "jpg", "jpeg"],
        key="pattern_uploader",
    )

generate_button = st.button("✨ 배경화면 생성하기")


# =========================
# 메인 로직
# =========================

if generate_button:
    if not uploaded_files:
        st.error("메인 이미지를 최소 1장 이상 업로드해 주세요.")
    else:
        pil_images = [Image.open(f).convert("RGB") for f in uploaded_files]

        col_left, col_right = st.columns(2)

        # 왼쪽: 원본 + 팔레트
        with col_left:
            st.subheader("① 업로드한 이미지들 (Moodboard)")
            for img in pil_images:
                st.image(img, use_column_width=True)

            with st.spinner("여러 이미지에서 공통 컬러 팔레트 추출 중..."):
                base_colors = extract_palette_from_images(
                    pil_images, num_palette_colors
                )
                adjusted_colors = adjust_colors_with_mood(
                    base_colors, brightness_level, saturation_level
                )
                palette_fig = plot_palette(adjusted_colors)

            st.subheader("② 무드 파라미터 반영된 컬러 팔레트")
            if palette_fig is not None:
                st.pyplot(palette_fig)
            else:
                st.write("팔레트를 추출할 수 없습니다.")

        # 오른쪽: 무드 설명 + 배경화면
        with col_right:
            st.subheader("③ 무드 설명 & 배경화면 생성")

            mood_param_text = describe_mood_params(
                brightness_level, saturation_level, abstract_level
            )
            st.markdown("**무드 파라미터 설명**")
            st.write(mood_param_text)

            st.markdown("**자동 무드 & 스타일 분석 (룰 기반)**")
            st.write(
                heuristic_mood_description(
                    adjusted_colors,
                    brightness_level,
                    saturation_level,
                    abstract_level,
                )
            )

            if "패턴 템플릿" in generation_mode:
                st.markdown("---")
                st.markdown("**선택한 패턴 템플릿 미리보기**")
                if pattern_file is not None:
                    st.image(Image.open(pattern_file), use_column_width=True)
                else:
                    st.info("패턴 템플릿 이미지를 업로드하면 여기 미리 보입니다.")

            st.markdown("---")
            st.subheader("④ 생성된 배경화면")

            with st.spinner("배경화면을 생성하는 중..."):
                wallpaper_img = None

                if generation_mode.startswith("단색"):
                    wallpaper_img = generate_solid_wallpaper(adjusted_colors)

                elif "Soft" in generation_mode:
                    wallpaper_img = generate_soft_from_original(
                        pil_images[0],
                        brightness_level,
                        saturation_level,
                    )

                elif "Abstract" in generation_mode:
                    wallpaper_img = generate_abstract_background(
                        adjusted_colors,
                        abstract_level,
                    )

                elif "패턴 템플릿" in generation_mode:
                    if pattern_file is None:
                        st.error("패턴 템플릿 이미지를 업로드해 주세요.")
                    else:
                        pattern_img = Image.open(pattern_file).convert("RGB")
                        wallpaper_img = generate_pattern_from_template(
                            pattern_img,
                            adjusted_colors,
                            brightness_level,
                            saturation_level,
                        )

                if wallpaper_img is not None:
                    buf = io.BytesIO()
                    wallpaper_img.save(buf, format="PNG")
                    wallpaper_bytes = buf.getvalue()

                    st.image(wallpaper_bytes, use_column_width=True)
                    st.download_button(
                        label="📥 배경화면 이미지 다운로드",
                        data=wallpaper_bytes,
                        file_name="wallpaper.png",
                        mime="image/png",
                    )
                else:
                    st.warning("배경화면을 생성하지 못했습니다.")
