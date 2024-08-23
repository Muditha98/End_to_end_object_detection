import sys, os
import shutil
import glob
from signLanguage.pipeline.training_pipeline import TrainPipeline
from signLanguage.exception import SignException
from signLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template,Response
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)



class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()
    return "Training Successfull!!" 







@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        # Decode the incoming image
        image = request.json['image']
        decodeImage(image, clApp.filename)
        
        # Run YOLOv5 detection on the decoded image
        yolo_command = "cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.25 --source ../data/inputImage.jpg --save-txt --save-conf"
        os.system(yolo_command)
        
        # Find the latest 'exp' folder where YOLOv5 stores the results
        exp_dir = max(glob.glob('yolov5/runs/detect/exp*'), key=os.path.getctime)
        results_file = os.path.join(exp_dir, "labels/inputImage.txt")
        output_image_path = os.path.join(exp_dir, "inputImage.jpg")
        
        predictions = []
        
        # Check if the results file exists and contains data
        if os.path.exists(results_file) and os.path.getsize(results_file) > 0:
            with open(results_file, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height, confidence = map(float, line.strip().split())
                    predictions.append({
                        "class_id": int(class_id),
                        "bbox": [x_center, y_center, width, height],
                        "confidence": confidence
                    })
        else:
            print("No detection results found.")
        
        # Encode the detected image to base64
        if os.path.exists(output_image_path):
            opencodedbase64 = encodeImageIntoBase64(output_image_path)
            encoded_image = opencodedbase64.decode('utf-8')
        else:
            encoded_image = None
            print("No image found at expected path.")
        
        # Create the response JSON
        result = {
            "image": encoded_image,
            "predictions": predictions
        }

    except ValueError as val:
        print("ValueError:", val)
        return Response("Value not found inside json data")
    
    except KeyError as key_err:
        print("KeyError:", key_err)
        return Response("Key value error, incorrect key passed")
    
    except Exception as e:
        print("Exception:", e)
        return Response("Invalid input or processing error")

    return jsonify(result)




@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        print("Starting live detection...")
        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.25 --source 0 --view-img")

        return "Camera starting!!" 

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    



if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8000)