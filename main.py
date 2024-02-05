from fastapi import FastAPI
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
 PDFInfoNotInstalledError,
 PDFPageCountError,
 PDFSyntaxError
)


import boto3
from io import BytesIO

from pydantic import BaseModel
import json

class PDFURL(BaseModel):
    pdf_url: str

class ImageURL(BaseModel):
    image_url: str

from pathlib import Path
from glob import glob
app=FastAPI()

@app.get("/")
async def root():
    return {"message":"Hello World from STOPS!!"}


# aws_access_key_id = 'YOUR_ACCESS_KEY_ID'
# aws_secret_access_key = 'YOUR_SECRET_ACCESS_KEY'
# region_name = 'us-east-1'  # Replace with your desired AWS region




@app.post("/")
async def pdftoimage(url: PDFURL):
    bucket_name = "stops"
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
    print(url.pdf_url)
    print(type(url.pdf_url))
    response = s3.get_object(Bucket=bucket_name, Key=url.pdf_url)
    pdf_content =  response['Body'].read()

    filename = url.pdf_url.rsplit("/")[-1]
    key = "/".join(url.pdf_url.rsplit("/")[:-1])

    if len(key) == 0:
        destination = filename.split(".pdf")[0] + ".png" 
    else:
       destination = key + '/' + filename.split(".pdf")[0] + ".png" 
    images = convert_from_bytes(pdf_content)[0] #first image only
    # for i, image in enumerate(images):
    #     fname = "image" + str(i) + ".png"
    #     image.save(fname, "PNG")
    # Save image to BytesIO object
    image_bytesio = BytesIO()
    images.save(image_bytesio, format='PNG')
    image_bytesio.seek(0)
    
    s3.upload_fileobj(image_bytesio, bucket_name, destination)
    return json.dumps({"key": destination})



import torch
from typing import List
from transformers import LiltForTokenClassification, LayoutLMv3Processor
from transformers import LayoutLMv3FeatureExtractor, LayoutLMv3TokenizerFast, AutoTokenizer, LayoutLMv3Processor
from PIL import Image
import json
from io import BytesIO
import requests

#for caching
from functools import lru_cache


def create_bounding_box(bbox_data):
    xs = []
    ys = []
    for x, y in bbox_data:
        xs.append(x)
        ys.append(y)

    left = int(min(xs))
    top = int(min(ys))
    right = int(max(xs))
    bottom = int(max(ys))

    return [left, top, right, bottom]

def scale_bounding_box(box, width_scale, height_scale):
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale)
    ]


@lru_cache(maxsize=1)
def load_model():
    model_id_second="SCUT-DLVCLab/lilt-roberta-en-base"
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False) # set
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_id_second, truncation=True)
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_id = "Sachinkafle365/STOPS-ML-OCR" #hugging face custom model 
    model = LiltForTokenClassification.from_pretrained(model_id)
    model = model.to(device)

    return model, processor, tokenizer
   


@app.post("/predict")
async def predict(url: ImageURL):
    
    model, processor, tokenizer = load_model()
    bucket_name = "stops"

    textract = boto3.client("textract", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
    response = textract.detect_document_text(
          Document={
                      'Bytes': b'bytes',
                      'S3Object': {
                          'Bucket': bucket_name,
                          'Name': url.image_url
                      }
    }
                                            )
    
    ocr_page = []
    for item in response['Blocks']:
        if item['BlockType'] == 'WORD':
            word_info = {
                'word': item['Text'],
                'bounding_box': [
                    item['Geometry']['BoundingBox']['Left'],
                    item['Geometry']['BoundingBox']['Top'],
                    item['Geometry']['BoundingBox']['Width'],
                    item['Geometry']['BoundingBox']['Height']
                ]
            }
            ocr_page.append(word_info)


    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
 
    response = s3.get_object(Bucket=bucket_name, Key=url.image_url)
    image_content = response['Body'].read()

    # Open image using PIL
    image = Image.open(BytesIO(image_content)).convert("RGB").resize((350,450))
        

    width, height = image.size

    width_scale = 150 / width
    height_scale = 150 / height

    words = []
    boxes = []
    for row in ocr_page:
                boxes.append(scale_bounding_box(row["bounding_box"], width_scale, height_scale))
                words.append(row["word"])


    inputs = processor(
                            image,
                            words,
                            boxes=boxes,
                            max_length=512,  # Maximum sequence length
                            padding="max_length",  # Padding strategy
                            truncation=True,  # Truncation strategy
                            return_tensors="pt"  # Return PyTorch tensors
                        )
  # for key, value in inputs.items():
  #   print(f"{key}: shape={value.shape}, dtype={value.dtype}")

  # Inputs contains token-level input_ids, attention_mask, bbox, and pixel_values
    
    input_ids = inputs["input_ids"]

    attention_mask = inputs["attention_mask"]
    bbox = inputs["bbox"]
    pixel_values = inputs["pixel_values"]
    del inputs["pixel_values"]
    outputs = model(**inputs)
    tokenizer = processor.tokenizer

    # Extract the logits
    logits = outputs.logits

    # Convert logits to labels
    predicted_labels = logits.argmax(-1)
    label_list = model.config.id2label
    original_data = [(tokenizer.convert_ids_to_tokens(input_id.item()), label_list[label.item()]) for input_id, label in zip(input_ids[0], predicted_labels[0])]
    #print(tokens_labels)
    

    MATTER_LENGTH = [7]
    
    #filtered_data = [(token, label) for token, label in original_data if label != 'O']
    
    name = ''
    for i in range(len(original_data)):
        if original_data[i][1] == "B-CLIENT_NAMES":
          name += ''.join(original_data[i][0])
        if original_data[i][1] == "I-CLIENT_NAMES":
          name += ''.join(original_data[i][0])
        if original_data[i][1] == "E-CLIENT_NAMES":
          name += ''.join(original_data[i][0])
          if original_data[i+1][1] != "E-CLIENT_NAMES":
            break
    
    name =  name.replace('Ġ', ' ')
    
    
    property_address = ''
    for i in range(len(original_data)):
        if original_data[i][1] == "B-PROPERTY_ADDRESS":
            property_address += ''.join(original_data[i][0])
        if original_data[i][1] == "I-PROPERTY_ADDRESS":
            property_address += ''.join(original_data[i][0])
        if original_data[i][1] == "S-PROPERTY_ADDRESS":
            property_address += ''.join(original_data[i][0])
        if original_data[i][1] == "E-PROPERTY_ADDRESS":
            property_address += ''.join(original_data[i][0])
        if original_data[i][1] != "E-PROPERTY_ADDRESS":
            end = i + 1
            continue

    
    property_address = property_address.replace('Ġ', ' ').replace('<pad>', '').replace('2 Acacia Avenue','')
    
    
    client_address = ''
    for i in range(len(original_data)):
        if original_data[i][1] == "B-CLIENT_ADDRESS":
          client_address += ''.join(original_data[i][0])
        if original_data[i][1] == "I-CLIENT_ADDRESS":
          client_address += ''.join(original_data[i][0])
        if original_data[i][1] == "E-CLIENT_ADDRESS":
          client_address += ''.join(original_data[i][0])
          if original_data[i+1][1] != "E-CLIENT_ADDRESS":
            break
    
    client_address =  client_address.replace('Ġ', ' ')
    
    
    
    category = ''
    for i in range(len(original_data)):
        if original_data[i][1] == "B-CATEGORY_NAME":
          category += ''.join(original_data[i][0])
        if original_data[i][1] == "I-CATEGORY_NAME":
          category += ''.join(original_data[i][0])
        if original_data[i][1] == "E-CATEGORY_NAME":
          category += ''.join(original_data[i][0])
          if original_data[i+1][1] != "E-CATEGORY_NAME":
            break
    
    category =  category.replace('Ġ', ' ')
    
    
    matter = ''
    end = 0
    for i in range(len(original_data)):
        if original_data[i][1] == "S-MATTER_NUMBER":
          matter += ''.join(original_data[i][0]).strip()
          matter = matter.replace('Ġ', '')
          end = i
          if len(matter.replace('Ġ', '')) == MATTER_LENGTH[0]:
            break
          if original_data[i+1][1] !=  "S-MATTER_NUMBER":
            break
          
    while ( len(matter)  < MATTER_LENGTH[0] and len(matter) > 1 and len(matter) != 7):
      matter+= ''.join(original_data[end+1][0]).strip()
      end += 1

    matter = matter.replace('Ġ', ' ')
        
    output =  json.dumps({"MATTER_NUMBER": matter, "PROPERTY_ADDRESS": property_address.lstrip(),
                            "CLIENT_NAMES": name.lstrip(), "CLIENT_ADDRESS": client_address.lstrip(), "CATEGORY_NAME":  category.lstrip()})

    filename = url.image_url.rsplit("/")[-1]
    key = "/".join(url.image_url.rsplit("/")[:-1])

    if len(key) == 0:
        destination = filename.split(".png")[0] + ".json" 
    else:
       destination = key + '/' + filename.split(".png")[0] + ".json" 

    s3.put_object(Body = output, Bucket = bucket_name, Key= destination)
    s3.delete_object(Bucket= bucket_name, Key=url.image_url)
    return {"messsage": "success"}

    
