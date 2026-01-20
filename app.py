from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import logging
import time

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def gallery():
    image_folder = os.path.join('static', 'images')
    images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    app.logger.debug(f"Loaded images: {images}")
    return render_template('index.html', images=images)

@app.route('/process', methods=['POST'])
def process_image():
    image_name = request.form.get('image_name')
    if not image_name:
        app.logger.error("No image_name provided")
        return jsonify({'error': '이미지를 선택하세요'}), 400

    # 이미지 경로
    input_path = os.path.join('static', 'images', image_name)
    output_folder = os.path.join('static', 'processed')
    os.makedirs(output_folder, exist_ok=True)

    # 결과 이미지 경로
    result_paths = [
        os.path.join('static', 'det', f"{os.path.splitext(image_name)[0]}{os.path.splitext(image_name)[1]}"),
        os.path.join('static', 'mask', f"{os.path.splitext(image_name)[0]}{os.path.splitext(image_name)[1]}"),
        os.path.join('static', 'results', f"{os.path.splitext(image_name)[0]}{os.path.splitext(image_name)[1]}")
    ]
    output_paths = [
        os.path.join(output_folder, f'demo_1_{image_name}'),
        os.path.join(output_folder, f'demo_2_{image_name}'),
        os.path.join(output_folder, f'demo_3_{image_name}')
    ]

    app.logger.debug(f"Input path: {input_path}")
    app.logger.debug(f"Result paths: {result_paths}")
    app.logger.debug(f"Output paths: {output_paths}")

    try:
        # 입력 이미지 로드
        img = cv2.imread(input_path)
        if img is None:
            app.logger.error(f"Failed to load image: {input_path}")
            return jsonify({'error': 'Failed to Load Image'}), 500

        processed_urls = []
        for i, (result_path, output_path) in enumerate(zip(result_paths, output_paths), 1):
            if not os.path.exists(result_path):
                app.logger.warning(f"Result image not found: {result_path}, using input image as fallback")
                # 테스트용: 결과 이미지가 없으면 입력 이미지를 복사
                result_image = img
            else:
                result_image = cv2.imread(result_path)
                if result_image is None:
                    app.logger.error(f"Failed to load result image: {result_path}")
                    return jsonify({'error': f'Failed to load result image {i}'}), 500

            cv2.imwrite(output_path, result_image)
            app.logger.debug(f"Image {i} processed and saved to: {output_path}")
            processed_urls.append(f'processed/demo_{i}_{image_name}')

        app.logger.debug(f"Returning processed URLs: {processed_urls}")
        return jsonify({'processed_images': processed_urls})
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': '이미지 처리 중 오류가 발생했습니다. 다시 시도해주세요.'}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
