import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import colorsys


# =========================
# 이미지/팔레트 유틸
# =========================

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_palette_from_images(images, num_colors: int = 5, max_samples: int = 8000):
    """
    여러(또는 한) 이미지에서 공통 팔레트 추출 (단순 '많이 나온 색'이 아니라
    k-means 비슷한 방식으로 색 덩어리들을 중심색으로 뽑아서,
    서로 다른 색들이 잘 분리되도록 함)
    """
    all_pixels = []

    # 1) 이미지들을 모아서 픽셀 리스트 만들기
    for image in images:
        img = image.convert("RGB")
        # 너무 크게 하면 계산이 느려져서 적당히 줄이기
        img = img.resize((220, 220))
        arr = np.array(img, dtype=np.float32) / 255.0  # 0~1 범위
        pixels = arr.reshape(-1, 3)
        all_pixels.append(pixels)

    if not all_pixels:
        return np.array([])

    pixels = np.vstack(all_pixels)  # (N, 3)

    # 2) 샘플 수가 너무 많으면 일부만 랜덤 샘플링
    n_pixels = pixels.shape[0]
    if n_pixels > max_samples:
        idx = np.random.choice(n_pixels, max_samples, replace=False)
        pixels = pixels[idx]

    # 3) 간단 k-means (직접 구현)으로 num_colors개 중심색 찾기
    k = min(num_colors, len(pixels))
    rng = np.random.default_rng(42)

    # 초기 중심: 픽셀 중에서 랜덤 선택
    centers = pixels[rng.choice(len(pixels), k, replace=False)]

    for _ in range(12):  # 12번 정도 반복
        # 각 픽셀이 어떤 중심에 가장 가까운지 할당
        dists = np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (N, k)
        labels = np.argmin(dists, axis=1)

        new_centers = []
        for j in range(k):
            cluster_pixels = pixels[labels == j]
            if len(cluster_pixels) == 0:
                # 비어 있는 클러스터는 기존 중심 유지
                new_centers.append(centers[j])
            else:
                new_centers.append(cluster_pixels.mean(axis=0))
        new_centers = np.stack(new_centers, axis=0)

        # 변화량이 거의 없으면 조기 종료
        if np.allclose(new_centers, centers, atol=1e-3):
            centers = new_centers
            break
        centers = new_centers

    # 4) 각 클러스터 크기(픽셀 개수) 기준으로 정렬: 많이 등장한 색을 앞에
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    centers = centers[order]

    # 값 범위 보정
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
    """팔레트 색들을 #RRGGBB 리스트로 변환 (설명용)"""
    hex_list = []
    for rgb in colors:
        r, g, b = (rgb * 255).astype(int)
        hex_list.append(f"#{r:02X}{g:02X}{b:02X}")
    return hex_list


# =========================
# 컬러 조정 (무드 파라미터 반영)
# =========================

def adjust_colors_with_mood(colors, brightness_level, saturation_level):
    """
    0~1 brightness / saturation 슬라이더 값을 이용해
    팔레트 색을 전체적으로 조정 (간단한 HLS 조정)
    """
    if colors.size == 0:
        return colors

    adjusted = []
    for rgb in colors:
        r, g, b = rgb
        h, l, s = colorsys.rgb_to_hls(r, g, b)

        # 밝기 조정
        l = (l * 0.5) + (brightness_level * 0.5)
        # 채도 조정
        s = (s * 0.4) + (saturation_level * 0.6)

        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        adjusted.append([r2, g2, b2])

    return np.clip(np.array(adjusted), 0.0, 1.0)


# =========================
# 패턴 & 단색 / 배경 생성
# =========================

def generate_solid_wallpaper(colors, size=(1024, 1792)):
    """팔레트의 대표 색으로 단색 배경 생성"""
    width, height = size
    img = Image.new("RGB", size)
    if colors.size == 0:
        color = (240, 240, 240)
    else:
        rgb = colors[0]  # 첫 번째 색 사용
        color = tuple((rgb * 255).astype(int))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, height], fill=color)
    return img

import numpy as np  # 이미 있으면 중복 X

def choose_pattern_colors(colors, mode="two_plus_neutral"):
    """
    패턴용으로 팔레트에서 쓸 색만 골라주는 함수.
    - two_plus_neutral : 무채색 + 메인 2색
    - one_plus_neutral : 무채색 + 메인 1색
    - two              : 메인 2색만
    """
    if colors.size == 0:
        # fallback
        if mode == "one_plus_neutral":
            return np.array([[0.92, 0.92, 0.92], [0.3, 0.3, 0.3]])
        else:
            return np.array([[0.92, 0.92, 0.92], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6]])

    cols = colors.reshape(-1, 3)
    main1 = cols[0]
    main2 = cols[1] if cols.shape[0] > 1 else cols[0]

    neutral = np.array([0.92, 0.92, 0.92])  # 살짝 따뜻한 무채색

    if mode == "two":
        return np.stack([main1, main2], axis=0)
    elif mode == "one_plus_neutral":
        return np.stack([neutral, main1], axis=0)
    elif mode == "two_plus_neutral":
        return np.stack([neutral, main1, main2], axis=0)
    else:
        return cols


def generate_stripe_pattern(colors, size=(1024, 1792)):
    """
    팔레트에서 2색 + 무채색만 뽑아서
    반복되는 세로 스트라이프 패턴 생성
    """
    pattern_colors = choose_pattern_colors(colors, mode="two_plus_neutral")

    width, height = size
    img = Image.new("RGB", size)
    draw = ImageDraw.Draw(img)

    num_bands = len(pattern_colors) * 3  # 더 얇고 반복 많은 스트라이프
    stripe_width = int(width / num_bands) if num_bands > 0 else width

    for i in range(num_bands):
        rgb = pattern_colors[i % len(pattern_colors)]
        x0 = i * stripe_width
        x1 = (i + 1) * stripe_width if i < num_bands - 1 else width
        color = tuple((rgb * 255).astype(int))
        draw.rectangle([x0, 0, x1, height], fill=color)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return img



def generate_check_pattern(colors, size=(1024, 1792)):
    """
    타탄 체크 느낌:
    - 배경: 무채색
    - 메인 2색: 세로/가로 스트라이프
    - 세로/가로 줄을 반투명으로 겹치면서 교차 부분이 진해지게
    """
    pattern_colors = choose_pattern_colors(colors, mode="two_plus_neutral")
    base_neutral = pattern_colors[0]
    c1 = pattern_colors[1]
    c2 = pattern_colors[2] if pattern_colors.shape[0] > 2 else pattern_colors[1]

    width, height = size

    base_color = tuple((base_neutral * 255).astype(int))
    img = Image.new("RGB", size, base_color)

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # 반복 간격 (타탄 패턴 모듈)
    rep_w = width / 8
    rep_h = height / 10

    c1_rgba_wide = tuple((c1 * 255).astype(int)) + (120,)   # 넓은 줄
    c2_rgba_thin = tuple((c2 * 255).astype(int)) + (170,)   # 얇은 줄

    # 세로 스트라이프
    for i in range(10):
        x_start = i * rep_w

        # 넓은 줄 (c1)
        x0 = int(x_start + rep_w * 0.1)
        x1 = int(x_start + rep_w * 0.55)
        draw.rectangle([x0, 0, x1, height], fill=c1_rgba_wide)

        # 얇은 줄 (c2)
        x2 = int(x_start + rep_w * 0.65)
        x3 = int(x_start + rep_w * 0.8)
        draw.rectangle([x2, 0, x3, height], fill=c2_rgba_thin)

    # 가로 스트라이프
    for j in range(12):
        y_start = j * rep_h

        # 넓은 줄 (c1)
        y0 = int(y_start + rep_h * 0.15)
        y1 = int(y_start + rep_h * 0.45)
        draw.rectangle([0, y0, width, y1], fill=c1_rgba_wide)

        # 얇은 줄 (c2)
        y2 = int(y_start + rep_h * 0.6)
        y3 = int(y2 + rep_h * 0.18)
        draw.rectangle([0, y2, width, y3], fill=c2_rgba_thin)

    # 합성 + 약간의 블러로 질감 정리
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img


def generate_dot_pattern(colors, dot_scale=1.0, size=(1024, 1792)):
    """
    도트 패턴:
    - 배경: 무채색
    - 도트: 팔레트에서 고른 1색 (또는 2색)
    - 위/아래 줄이 엇갈리는 패턴
    - dot_scale로 크기 조절
    """
    pattern_colors = choose_pattern_colors(colors, mode="one_plus_neutral")
    neutral = pattern_colors[0]
    main = pattern_colors[1]

    width, height = size
    bg_color = tuple((neutral * 255).astype(int))
    dot_color = tuple((main * 255).astype(int))

    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)

    num_rows = 12
    num_cols = 7

    # 기본 반지름을 dot_scale로 조정
    base_radius = min(width / (num_cols * 3.5), height / (num_rows * 3.5))
    radius = int(base_radius * dot_scale)

    for row in range(num_rows):
        for col in range(num_cols):
            # 홀수 줄은 반 칸 offset → 엇갈리는 도트
            offset_x = radius if row % 2 == 1 else 0

            cx = int((col + 0.5) * width / num_cols) + offset_x
            cy = int((row + 0.5) * height / num_rows)

            # 화면 바깥으로 나간 점은 건너뛰기
            if cx + radius < 0 or cx - radius > width:
                continue

            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=dot_color,
            )

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def generate_soft_mood_background(colors, size=(1024, 1792)):
    """
    비슷한 무드의 부드러운 배경:
    위아래 그라디언트 + 반투명 컬러 덩어리 + 블러
    """
    if colors.size == 0:
        colors = np.array([[0.8, 0.8, 0.85], [0.9, 0.9, 0.95]])

    height, width = size[1], size[0]

    if len(colors) == 1:
        top = bottom = colors[0]
    else:
        top = colors[0]
        bottom = colors[-1]

    # 세로 그라디언트
    gradient = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        t = y / (height - 1)
        gradient[y, :, :] = (1 - t) * top + t * bottom

    gradient_uint8 = (gradient * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(gradient_uint8, mode="RGB")

    # 부드러운 컬러 덩어리 (블롭)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    num_blobs = 20
    for i in range(num_blobs):
        rgb = colors[i % len(colors)]
        base = np.array(rgb) * 255
        alpha = 80
        color = (int(base[0]), int(base[1]), int(base[2]), alpha)
        radius = np.random.randint(int(width * 0.1), int(width * 0.3))
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=6))
    return img


def generate_abstract_background(colors, abstract_level, size=(1024, 1792)):
    """
    팔레트 색을 사용해서 수채화 느낌의 추상 배경 생성:
    - 팔레트의 두 색으로 세로 그라디언트 깔고
    - 반투명한 '물감 블롭'들을 여러 겹으로 얹은 뒤
    - 전체를 블러 + 살짝 그레인 추가
    """
    import numpy as np, random
    from PIL import Image, ImageDraw, ImageFilter

    # 팔레트가 비어 있을 때 대비용 기본 색
    if colors.size == 0:
        colors = np.array([
            [0.82, 0.82, 0.88],
            [0.35, 0.40, 0.55],
            [0.93, 0.86, 0.80],
        ])

    width, height = size

    # 1) 팔레트에서 위/아래 그라디언트용 두 색 선택
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

    # 2) 수채화처럼 번지는 반투명 블롭들
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    base_blobs = 25          # 기본 블롭 개수
    extra = int(abstract_level * 45)  # 추상 정도에 따라 추가
    num_blobs = base_blobs + extra

    for i in range(num_blobs):
        rgb = colors[i % len(colors)]
        r, g, b = (rgb * 255).astype(int)

        # 반투명 알파
        alpha = random.randint(40, 110)
        color = (r, g, b, alpha)

        # 블롭 크기 (추상 정도에 따라 더 크게)
        max_radius = int(min(w, h) * (0.25 + 0.25 * abstract_level))
        min_radius = int(min(w, h) * 0.08)
        rx = random.randint(min_radius, max_radius)
        ry = int(rx * random.uniform(0.6, 1.4))

        # 위치는 화면 주변까지 넓게 랜덤
        cx = random.randint(-int(w * 0.1), int(w * 1.1))
        cy = random.randint(-int(h * 0.1), int(h * 1.1))

        # wobble 느낌: 살짝씩 흔들린 타원 여러 번 겹쳐 그림
        jitter_times = random.randint(2, 4)
        for _ in range(jitter_times):
            jx = int(cx + random.uniform(-rx * 0.15, rx * 0.15))
            jy = int(cy + random.uniform(-ry * 0.15, ry * 0.15))
            draw.ellipse([jx - rx, jy - ry, jx + rx, jy + ry], fill=color)

    # 3) 그라디언트 배경 + 블롭 합성
    composed = Image.alpha_composite(img.convert("RGBA"), overlay)

    # 4) 수채화 번짐처럼 전체 블러
    blur_radius = 5 + abstract_level * 4  # 추상 정도 높을수록 더 흐릿하게
    composed = composed.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 5) 아주 약한 그레인(노이즈) 추가해서 디지털 티 조금 줄이기
    arr = np.array(composed.convert("RGB")).astype("int16")
    noise_strength = 12
    noise = np.random.randint(-noise_strength, noise_strength + 1, size=arr.shape[:2] + (1,))
    arr = np.clip(arr + noise, 0, 255).astype("uint8")

    final_img = Image.fromarray(arr, mode="RGB")
    return final_img


# =========================
# 무드 설명 (룰 기반)
# =========================

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
    """팔레트와 슬라이더 값을 이용해 간단 무드 설명 생성"""
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

    # 대충 hue 기준으로 warm/cool 판별
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
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Moodboard 월페이퍼 생성기 (로컬)",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Moodboard 기반 월페이퍼 생성기 (OpenAI API 없음)")
st.write(
    """
이미지 **1장 또는 여러 장**을 업로드하면,  
공통된 **무드 & 컬러 팔레트**를 분석해서  
선택한 방식으로 **배경화면(단색 / 비슷한 무드 느낌 / 추상 / stripe / check / dot)**를 생성합니다.  
모든 계산은 로컬 알고리즘으로만 진행됩니다.
"""
)

# 사이드바
generation_mode = st.sidebar.selectbox(
    "배경화면 타입 선택",
    [
        "단색 (Solid color)",
        "비슷한 무드의 이미지 느낌 (Soft)",
        "추상 배경화면 (Abstract)",
        "Stripe 패턴",
        "Check 패턴",
        "Dot 패턴",
    ],
)

num_palette_colors = st.sidebar.slider("팔레트 색상 개수", 3, 8, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("무드 파라미터")
brightness_level = st.sidebar.slider("Brightness (밝기)", 0.0, 1.0, 0.6, 0.05)
saturation_level = st.sidebar.slider("Saturation (채도)", 0.0, 1.0, 0.7, 0.05)
abstract_level = st.sidebar.slider("Abstractness (추상 정도)", 0.0, 1.0, 0.7, 0.05)

# 🔹 도트 크기 슬라이더 추가
dot_scale = st.sidebar.slider("Dot size scale (도트 크기)", 0.5, 2.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.write("1. 이미지 업로드 (1장 또는 여러 장) → 2. 생성 버튼 클릭")

# 메인 영역
uploaded_files = st.file_uploader(
    "무드를 만들 이미지를 업로드하세요 (1장 또는 여러 장, 룩북, OOTD, 레퍼런스 등)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

generate_button = st.button("✨ 배경화면 생성하기")

if generate_button:
    if not uploaded_files:
        st.error("이미지를 최소 1장 이상 업로드해 주세요.")
    else:
        # 1장이어도 리스트로 처리 가능
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

            st.markdown("---")
            st.subheader("④ 생성된 배경화면")

            with st.spinner("배경화면을 생성하는 중..."):
                wallpaper_img = None

                if generation_mode.startswith("단색"):
                    wallpaper_img = generate_solid_wallpaper(adjusted_colors)
                elif "Soft" in generation_mode:
                    wallpaper_img = generate_soft_mood_background(adjusted_colors)
                elif "Abstract" in generation_mode:
                    wallpaper_img = generate_abstract_background(
                        adjusted_colors, abstract_level
                    )
                elif generation_mode.startswith("Stripe"):
                    wallpaper_img = generate_stripe_pattern(adjusted_colors)
                elif generation_mode.startswith("Check"):
                    wallpaper_img = generate_check_pattern(adjusted_colors)
                elif generation_mode.startswith("Dot"):
                    wallpaper_img = generate_dot_pattern(adjusted_colors, dot_scale=dot_scale)

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
