import cv2
import pytesseract
import os
import json

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\\tesseract.exe'

def process_image(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plates = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        if w > 100 and h > 30 and w / h < 6:
            roi = image[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi, config='--psm 7').strip()

            if text:
                plates.append({
                    "box": [x, y, w, h],
                    "text": text
                })
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return plates, image

def main():
    image_folder = 'images'
    output = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            path = os.path.join(image_folder, filename)
            plates, image_with_boxes = process_image(path)

            output.append({
                "filename": filename,
                "plates": plates
            })

            cv2.imwrite(f"out_{filename}", image_with_boxes)

    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Готово!")

if __name__ == '__main__':
    main()
