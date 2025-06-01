import cv2
import pytesseract
import os
import json

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\\tesseract.exe'

plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

def clean_plate_text(text):
    text = text.replace(" ", "").replace("\n", "").upper()
    return text

def process_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plates_detected = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 30))

    plates = []

    for (x, y, w, h) in plates_detected:
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config='--psm 7')
        text = clean_plate_text(text)

        if text:
            plates.append({
                "box": [int(x), int(y), int(w), int(h)],
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
