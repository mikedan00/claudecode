"""이미지 지각 해시(dHash) 기반 유사도. 크기 변경·재압축·약한 보정에도 견고하다."""
import io

from PIL import Image


def dhash(data: bytes, size: int = 16) -> int:
    img = Image.open(io.BytesIO(data)).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(img.tobytes())
    bits = 0
    for row in range(size):
        for col in range(size):
            left, right = px[row * (size + 1) + col], px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def hash_similarity(a: int, b: int, size: int = 16) -> float:
    """0~1, 1이면 동일. 일반적으로 0.9 이상이면 같은 이미지에서 파생된 것으로 본다."""
    return 1 - bin(a ^ b).count("1") / (size * size)
