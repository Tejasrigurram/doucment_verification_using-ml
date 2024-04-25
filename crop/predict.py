import os
from ultralytics import YOLO
import cv2



image_filename = '29.jpg'
image_path = 'New folder/ps/images/29.png'
OUTPUT_DIR = 'New folder/ps/result'
output_image_path = os.path.join(OUTPUT_DIR, '{}_out.jpg'.format(os.path.splitext(image_filename)[0]))

# Load an image
frame = cv2.imread(image_path)
H, W, _ = frame.shape

model_path = 'New folder/ps/runs/detect/train/weights/best.pt'

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

# Perform inference on the image
results = model(frame)[0]
print(results.boxes.data.tolist())
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:

        cropped_object = frame[int(y1):int(y2), int(x1):int(x2)]

        # Save the cropped object
        output_cropped_path = os.path.join(OUTPUT_DIR, f'{results.names[int(class_id)].lower()}_crop.jpg')
        cv2.imwrite(output_cropped_path, cropped_object)


        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Save the output image
cv2.imwrite(output_image_path, frame)
print("//////////result/////////////////////")
print("we have found the ",results.names[int(class_id)]," document")
