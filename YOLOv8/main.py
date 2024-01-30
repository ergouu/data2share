import base64
import io
import json

import yaml
from model_handler import ModelHandler
from PIL import Image


def init_context(context):
    context.logger.info("Init context...  0%")

    # Read labels
    with open("/opt/nuclio/function-gpu.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)
    context.logger.info("Init context...  1%")

    labels_spec = functionconfig['metadata']['annotations']['spec']
    context.logger.info(labels_spec)
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}
    context.logger.info(labels)
    context.logger.info("Init context...  2%")

    # Read the DL model
    model = ModelHandler(labels)
    context.logger.info("Init context...  3%")
    context.user_data.model = model

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run YoloV8 tensorflow model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf)

    results = context.user_data.model.infer(image, threshold)

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)