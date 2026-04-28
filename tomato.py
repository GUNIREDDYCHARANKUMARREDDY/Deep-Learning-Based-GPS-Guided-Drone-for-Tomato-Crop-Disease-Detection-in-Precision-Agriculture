import time
import csv
import sys
import signal
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
from picamera2 import Picamera2

# GPS
import serial
import pynmea2

# ==================== CONFIG ====================
MODEL_PATH = "tomato_resnet18.pth"
DEVICE = torch.device("cpu")

CAPTURE_INTERVAL = 2.0
FRAME_SIZE = (320, 240)

class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

treatment_recommendations = {
    "Tomato___Bacterial_spot": {"Chemical": "Plantomycin", "Product": "Streptomycin + Tetracycline"},
    "Tomato___Early_blight": {"Chemical": "Amister", "Product": "Azoxystrobin + Difenoconazole"},
    "Tomato___Late_blight": {"Chemical": "RANMAN", "Product": "Cyazofamid"},
    "Tomato___Leaf_Mold": {"Chemical": "SULTAF", "Product": "Sulphur"},
    "Tomato___Septoria_leaf_spot": {"Chemical": "Kavach", "Product": "Chlorothalonil"},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"Chemical": "Delegate", "Product": "Spinetoram"},
    "Tomato___Target_Spot": {"Chemical": "INDOFIL M-45", "Product": "Mancozeb"},
    "Tomato___Tomato_mosaic_virus": {"Chemical": "ORGA NEEM", "Product": "Neem"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"Chemical": "Actara", "Product": "Thiamethoxam"},
    "Tomato___healthy": {"Chemical": "RALLIS BAHAAR", "Product": "Amino acids"}
}

CSV_PATH = "live_predictions.csv"

# ==================== MODEL ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(class_names))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==================== GPS ====================
gps_serial = serial.Serial("/dev/serial0", 9600, timeout=1)

def get_gps_data():
    try:
        line = gps_serial.readline().decode('ascii', errors='replace')

        if line.startswith('$GPRMC'):
            msg = pynmea2.parse(line)

            # ONLY valid GPS data
            if msg.status == 'A':
                return msg.latitude, msg.longitude

    except:
        pass

    return None, None

# ==================== CAMERA ====================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": FRAME_SIZE}))
picam2.start()
time.sleep(1)

# ==================== CSV ====================
csv_file = open(CSV_PATH, mode='w', newline='')
writer = csv.writer(csv_file)
writer.writerow(["Timestamp", "Class", "Latitude", "Longitude", "Chemical", "Product"])

# ==================== CLEANUP ====================
def cleanup(sig, frame):
    picam2.stop()
    cv2.destroyAllWindows()
    csv_file.close()
    print("\nSaved to CSV")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

# ==================== LOOP ====================
last_time = 0

print("🚀 Live Tomato Detection + GPS Started")

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    display_text = "Waiting for detection..."

    if time.time() - last_time > CAPTURE_INTERVAL:
        last_time = time.time()

        # Prediction
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            out = model(img)
            pred = torch.argmax(out, 1).item()
            display_text = class_names[pred]

        # GPS
        lat, lon = get_gps_data()

        # Write ONLY if GPS valid
        if lat is not None and lon is not None:
            treatment = treatment_recommendations.get(display_text, {"Chemical": "N/A", "Product": "N/A"})

            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                display_text,
                lat,
                lon,
                treatment["Chemical"],
                treatment["Product"]
            ])
            csv_file.flush()

    # ==================== DISPLAY ====================
    cv2.putText(frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Tomato Detection + GPS", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cleanup(None, None)