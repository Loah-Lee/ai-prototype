import numpy as np
from PIL import Image

DRAW_THRESHOLD = 20
TARGET_SIZE = 28
INNER_SIZE = 20
BOUNDING_BOX_PADDING = 12


def prepare_canvas_for_inference(image_data: np.ndarray) -> dict[str, object]:
    """캔버스 RGBA 이미지를 모델 입력용 28x28 텐서로 전처리한다."""
    if image_data is None:
        raise ValueError("캔버스에 숫자를 먼저 그려 주세요.")

    rgba_array = np.clip(image_data, 0, 255).astype(np.uint8)
    rgba_image = Image.fromarray(rgba_array, mode="RGBA")
    background = Image.new("RGBA", rgba_image.size, (0, 0, 0, 255))
    grayscale_image = Image.alpha_composite(background, rgba_image).convert("L")

    grayscale_array = np.asarray(grayscale_image)
    if np.count_nonzero(grayscale_array > DRAW_THRESHOLD) < 25:
        raise ValueError("숫자가 충분히 그려지지 않았습니다. 다시 입력해 주세요.")

    top, bottom, left, right = _find_bounding_box(grayscale_array)
    cropped_array = grayscale_array[top : bottom + 1, left : right + 1]
    cropped_image = Image.fromarray(cropped_array, mode="L")
    resized_image = _resize_to_fit(cropped_image, INNER_SIZE)

    preprocessed_image = Image.new("L", (TARGET_SIZE, TARGET_SIZE), color=0)
    offset_x = (TARGET_SIZE - resized_image.width) // 2
    offset_y = (TARGET_SIZE - resized_image.height) // 2
    preprocessed_image.paste(resized_image, (offset_x, offset_y))

    input_tensor = np.asarray(preprocessed_image, dtype=np.float32) / 255.0
    input_tensor = input_tensor[np.newaxis, np.newaxis, :, :]

    return {
        "original_image": grayscale_image,
        "preprocessed_image": preprocessed_image,
        "input_tensor": input_tensor,
    }


def _find_bounding_box(image_array: np.ndarray) -> tuple[int, int, int, int]:
    positions = np.argwhere(image_array > DRAW_THRESHOLD)
    if positions.size == 0:
        raise ValueError("캔버스에 숫자를 먼저 그려 주세요.")

    top, left = positions.min(axis=0)
    bottom, right = positions.max(axis=0)

    top = max(int(top) - BOUNDING_BOX_PADDING, 0)
    left = max(int(left) - BOUNDING_BOX_PADDING, 0)
    bottom = min(int(bottom) + BOUNDING_BOX_PADDING, image_array.shape[0] - 1)
    right = min(int(right) + BOUNDING_BOX_PADDING, image_array.shape[1] - 1)
    return top, bottom, left, right


def _resize_to_fit(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    if width == 0 or height == 0:
        raise ValueError("전처리 중 빈 이미지가 생성되었습니다.")

    if width >= height:
        new_width = max_side
        new_height = max(1, round(height * max_side / width))
    else:
        new_height = max_side
        new_width = max(1, round(width * max_side / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
