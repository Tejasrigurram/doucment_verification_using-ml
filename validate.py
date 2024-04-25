import os
from ultralytics import YOLO
import cv2

image_path = 'result/aadhar_card_crop.jpg'
OUTPUT_DIR = os.path.join('.', 'validate')
output_image_path = os.path.join(OUTPUT_DIR, 'output.jpg')

# Load an imaged
oframe = cv2.imread(image_path)
frame = cv2.resize(oframe, (420,640  ))
H, W, _ = frame.shape

model_path = 'adharmodel/best2.pt'

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.4

# Perform inference on the image
results = model(frame)[0]

# List to track detected classes
detected_classes = []
print(results.boxes.data.tolist())
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        detected_classes.append(results.names[int(class_id)])
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Print information for each detected box
        print("Detected:", results.names[int(class_id)], " with confidence:", score)

# Save the output image
cv2.imwrite(output_image_path, frame)

# Specify the classes to check for detection
specified_classes = ['aadhar_QR', 'aadhar_holder_photo', 'aadhar_number', 'aadhar_logo',
                     'emblem_of_india', 'aadhar_holder_DOB', 'aadhar_holder_gender', 'aadhar_holder_name']

# Check if all specified classes are detected
missing_classes = [cls for cls in specified_classes if cls not in detected_classes]

if not missing_classes:
    print("Verified: All specified classes are detected.")
else:
    print("Not Verified: Some specified classes are not detected.")
    print("Missing classes:", missing_classes)



#/ / / / / / 

# Example standard coordinates (adjust these values based on your Aadhar card layout)
# standard_coordinates = {
#     'aadhar_number': [122.0924072265625, 493.30133056640625, 313.6906433105469, 558.2333374023438],
#     'aadhar_holder_name': [121.08172607421875, 191.08306884765625, 317.03338623046875, 235.2215576171875],
#     'aadhar_QR': [314.52044677734375, 329.9539794921875, 411.20013427734375, 545.97412109375],
#     'aadhar_card': [0.0, 10.39874267578125, 420.0, 637.628173828125],
#     'aadhar_logo': [118.49209594726562, 57.975616455078125, 355.0701599121094, 162.11886596679688],
#     'emblem_of_india': [47.563175201416016, 53.9162483215332, 93.51278686523438, 158.0561065673828],
#     'aadhar_holder_photo': [23.711009979248047, 158.40301513671875, 125.61561584472656, 449.15625],
#     'aadhar_holder_DOB': [124.84255981445312, 233.8388671875, 299.17901611328125, 278.5816650390625],
#     'aadhar_holder_gender': [122.1571044921875, 271.14776611328125, 226.12249755859375, 313.0513916015625],
# }

# / // / / / / / / / ///  / / / / / / / 


# Replace hypothetical coordinates with actual coordinates from standard_coordinates
specified_classes_with_positions = {
    'aadhar_holder_name': (121.08172607421875, 191.08306884765625),
    'aadhar_holder_DOB': (124.84255981445312, 233.8388671875),
    'aadhar_holder_gender': (122.1571044921875, 271.14776611328125),
    'aadhar_QR': (314.52044677734375, 329.9539794921875),
    'aadhar_holder_photo': (23.711009979248047, 158.40301513671875),
    'emblem_of_india': (47.563175201416016, 53.9162483215332),
    # Add more classes as needed
}
# Thresholds for relative positions
threshold_x_relative = 100
threshold_y_relative = 100

# Check if all specified classes are detected and correctly aligned relative to each other
correctly_aligned_classes_relative = []
incorrectly_aligned_classes_relative = []

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        detected_class = results.names[int(class_id)]

        # Check if the detected class is one of the specified classes
        if detected_class in specified_classes_with_positions:
            # Get the specified relative position for the detected class
            specified_position = specified_classes_with_positions[detected_class]

            # Calculate the expected coordinates based on relative positions
            expected_x = specified_position[0] 
            expected_y = specified_position[1] 

            # Check if the detected coordinates are within a threshold of the expected coordinates
            if (
                abs(x1 - expected_x) <= threshold_x_relative
                and abs(y1 - expected_y) <= threshold_y_relative
            ):
                correctly_aligned_classes_relative.append(detected_class)
            else:
                incorrectly_aligned_classes_relative.append(detected_class)

                # Print information for each incorrectly aligned box
                print("Incorrectly aligned (relative):", detected_class, " with confidence:", score)

# Print the verification result based on relative positions
if not incorrectly_aligned_classes_relative:
    print("Verified (relative): All specified classes are detected and correctly aligned.")
else:
    print("Not Verified (relative): Some specified classes are not correctly aligned.")
    print("Incorrectly aligned classes (relative):", incorrectly_aligned_classes_relative)
    print("NOT AN VALID AADHAR CARD")