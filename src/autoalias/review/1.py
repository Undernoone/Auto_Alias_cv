from PIL import Image
import os

def split_vertical_9(input_path, output_dir="output"):
    img = Image.open(input_path)
    width, height = img.size

    os.makedirs(output_dir, exist_ok=True)

    num_parts = 9

    for i in range(num_parts):
        left = 0
        right = width

        upper = i * height // num_parts
        lower = (i + 1) * height // num_parts

        cropped = img.crop((left, upper, right, lower))

        output_path = os.path.join(output_dir, f"part_{i + 1}.png")
        cropped.save(output_path)

        print(f"保存: {output_path}, 尺寸: {cropped.size}")

if __name__ == "__main__":
    split_vertical_9(r"F:\430AutoAlias\src\autoalias\review\12.png")