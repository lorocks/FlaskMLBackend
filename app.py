import os
import base64
from flask import Flask, request, render_template, redirect, url_for, jsonify
from supabase import create_client, Client
from skimage.segmentation import clear_border
import datetime
import numpy as np
import cv2
# from PIL import Image
# import json
# import tensorflow as tf #might remove
# from object_detection.utils import visualization_utils as viz_utils
# from easyocr import Reader
from paddleocr import PaddleOCR
import torch

model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'YOLO/best.pt')
# model = torch.hub.load('yolov7', 'custom', 'YOLO/best.pt', source='local')
# detect_fn = tf.saved_model.load("saved_model/export_test/saved_model")
# category_index = {1: {'id': 1, 'name': 'licence'}}

# easyReader = Reader(["en"])
paddleReader = PaddleOCR(use_angle_cls=False, lang="en", det_model_dir="whl/det/en/en_PP-OCRv3_det_infer",
                         rec_model_dir="whl/rec/en/en_PP-OCRv3_rec_infer", cls_model_dir="whl/cls/ch_ppocr_mobile_v2.0_cls_infer")

app = Flask(__name__)

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ANONKEY")
secret = os.environ.get("SECRET_KEY")

supabase: Client = create_client(url, key)
@app.route("/")
def index():
    print(url, key, secret)
    return f"Flask App"


@app.route("/sendimage")
def image():
    if request.method == 'GET': #change to POST
        fileimage = request.get_json()['image']
        getSecret = request.get_json()['secret']

        if secret == getSecret:
            photo = base64.b64decode(fileimage)
            uniqueId = datetime.datetime.now().timestamp()
            idNum = len(list(supabase.table("images").select("*").execute())[0][1])
            imagename = f'{uniqueId}.png'
            data = {
                "id": idNum+1,
                "imageName": imagename,
            }
            resp = supabase.table("images").insert(data).execute()
            with open(imagename, "wb") as file:
                res = supabase.storage().from_("test").upload(f"/images/{imagename}", photo)
            try:
                os.remove(imagename)
            except:
                return "Image saved but no delete :<"
            return "Image has been saved"
        return "Secret failed"
    else:
        return "Smh use propoerly"

# @app.route("/sendlicenseT")
# def licenseT():
#     if request.method == 'GET': #change to POST
#         fileimage = request.get_json()['image']
#         getSecret = request.get_json()['secret']
#
#         if secret == getSecret:
#             photo = base64.b64decode(fileimage)
#             np_data = np.frombuffer(photo, np.uint8)
#             image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
#             image_np = np.array(image)
#             input_tensor = tf.convert_to_tensor(image_np)
#             input_tensor = input_tensor[tf.newaxis, ...]
#             detections = detect_fn(input_tensor)
#
#             try:
#                 if np.amax(detections["detection_scores"]) < 0.5:
#                     return "Detections confidence too low"
#             except:
#                 return "No Detections"
#
#             num_detections = int(detections.pop('num_detections'))
#             detections = {key: value[0, :num_detections].numpy()
#                           for key, value in detections.items()}
#             detections['num_detections'] = num_detections
#
#             detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
#
#             image_np_with_detections = image_np.copy()
#
#             boxes = viz_utils.visualize_boxes_and_labels_on_image_array(
#                 image_np_with_detections,
#                 detections['detection_boxes'],
#                 detections['detection_classes'],
#                 detections['detection_scores'],
#                 category_index,
#                 use_normalized_coordinates=True,
#                 max_boxes_to_draw=200,
#                 min_score_thresh=.5,
#                 agnostic_mode=False)
#
#             ret, buffer = cv2.imencode('.jpg', image_np_with_detections)
#             middle_image = base64.b64encode(buffer).decode('utf-8')
#             final_image = base64.b64decode(middle_image)
#
#             # bounding_boxes = list(boxes[1].keys())
#             # recognition = []
#             for i, thing in enumerate(detections["detection_boxes"]):
#                 if detections["detection_scores"][i] < 0.5:
#                     break
#                 cropped = cv2.resize(image_np, (320, 320))
#                 ymin = int(max(1, thing[0] * 320)) - 5
#                 xmin = int(max(1, thing[1] * 320)) - 5
#                 ymax = int(min(320, thing[2] * 320)) + 5
#                 xmax = int(min(320, thing[3] * 320)) + 5
#                 cropped = cropped[ymin:ymax, xmin:xmax, ...]
#                 test_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#                 th3 = cv2.bilateralFilter(test_img, 11, 17, 17)
#                 roi_image = cv2.threshold(th3, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#                 roi_image = clear_border(roi_image)
#                 roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
#                 test_img1 = cv2.fastNlMeansDenoisingColored(roi_image, None, 20, 20, 7, 21)
#                 test_img1 = cv2.fastNlMeansDenoisingColored(test_img1, None, 20, 7, 21)
#                 result = paddleReader.ocr(test_img1, cls=False)
#                 text = ""
#                 for predict in result[0]:
#                     text = text + predict[1][0] + " "
#                 if text == "":
#                     result = paddleReader.ocr(cropped, cls=False)
#                     text = ""
#                     for predict in result[0]:
#                         text = text + predict[1][0] + " "
#
#                 uniqueId = datetime.datetime.now().timestamp()
#                 idNum = len(list(supabase.table("LPDetection").select("*").execute())[0][1])
#                 imagename = f'{uniqueId}.jpg'
#                 data = {
#                     "id": idNum+1,
#                     "imageName": imagename,
#                     "LPOCR": text,
#                     "Fined": False
#                 }
#                 resp = supabase.table("LPDetection").insert(data).execute()
#                 with open(imagename, "wb") as file:
#                     res = supabase.storage().from_("licenses").upload(f"/images/{imagename}", final_image)
#                 try:
#                     os.remove(imagename)
#                 except:
#                     return "Image saved but no delete :<"
#             return "Image has been saved"
#     else:
#         return "Smh use properly"

@app.route("/sendlicenseY")
def licenseY():
    if request.method == 'GET': #change to POST
        fileimage = request.get_json()['image']
        getSecret = request.get_json()['secret']

        if secret == getSecret:
            photo = base64.b64decode(fileimage)
            np_data = np.frombuffer(photo, np.uint8)
            image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
            image_np = np.array(image)

            results = model(image_np)
            detections = results.pandas().xyxy[0]
            if len(detections) == 0:
                return "No Detections"
            elif detections.iloc[0].confidence < 0.5:
                return "Detection confidence too low"

            rows = detections.shape[0]
            image_np_with_detections = image_np.copy()
            for i in range(rows):
                xmin = int(detections.iloc[i].xmin)
                xmax = int(detections.iloc[i].xmax)
                ymin = int(detections.iloc[i].ymin)
                ymax = int(detections.iloc[i].ymax)

                cv2.rectangle(image_np_with_detections, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                cv2.putText(image_np_with_detections, str(round(results.pandas().xyxy[0].confidence.get(i), 4)), (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', image_np_with_detections)
                middle_image = base64.b64encode(buffer).decode('utf-8')
                final_image = base64.b64decode(middle_image)

                cropped = image_np[ymin:ymax, xmin:xmax, ...]
                test_img = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                th3 = cv2.bilateralFilter(test_img, 11, 17, 17)
                roi_image = cv2.threshold(th3, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                roi_image = clear_border(roi_image)
                roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
                test_img1 = cv2.fastNlMeansDenoisingColored(roi_image, None, 20, 20, 7, 21)
                test_img1 = cv2.fastNlMeansDenoisingColored(test_img1, None, 20, 7, 21)
                result = paddleReader.ocr(test_img1, cls=False)
                text = ""
                for predict in result[0]:
                    text = text + predict[1][0] + " "
                if text == "":
                    result = paddleReader.ocr(cropped, cls=False)
                    text = ""
                    for predict in result[0]:
                        text = text + predict[1][0] + " "

                uniqueId = datetime.datetime.now().timestamp()
                idNum = len(list(supabase.table("LPDetection").select("*").execute())[0][1])
                imagename = f'{uniqueId}.jpg'
                data = {
                    "id": idNum+1,
                    "imageName": imagename,
                    "LPOCR": text,
                    "Fined": False
                }
                resp = supabase.table("LPDetection").insert(data).execute()
                with open(imagename, "wb") as file:
                    res = supabase.storage().from_("licenses").upload(f"/images/{imagename}", final_image)
                try:
                    os.remove(imagename)
                except:
                    return "Image saved but no delete :<"
            return "Image has been saved"
        return "Secret failed"
    else:
        return "Smh use properly"

@app.route("/test")
def test():
    idNum = len(list(supabase.table("profiles").select("*").execute())[0][1])
    data = {
        "id": idNum+1,
        "username": "test",
        "password": "Test123$"
    }
    res = supabase.table("profiles").insert(data).execute()
    return "Ok brooo"

@app.route("/<id>")
def default(id):
    return f"Why did you type {id}?"

if __name__=='__main__':
    app.run(debug=True, host="192.168.10.149", port=5001)
