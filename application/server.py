# server.py

from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
from urllib.parse import urlencode
from fastapi import FastAPI, Lifespan


import cv2
import numpy as np

import torch
import base64
import random
import json
import sqlite3
from datetime import datetime

app = FastAPI()
templates = Jinja2Templates(directory='templates')

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set default model
DEFAULT_MODEL_NAME = 'yolov5m'
model = None  # Will hold the loaded model

# Color palette for bounding boxes
colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)]  # for bbox plotting

# Database setup
DB_PATH = 'logs.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def add_column_if_not_exists(conn, table, column, column_type):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [info[1] for info in cursor.fetchall()]
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
        conn.commit()
        print(f"Added column '{column}' to table '{table}'.")

# Load the model at startup
@app.on_event("startup")
def load_model_and_setup_db():
    global model
    model = torch.hub.load('ultralytics/yolov5', DEFAULT_MODEL_NAME, pretrained=True)
    model.eval()
    print(f"Loaded model: {DEFAULT_MODEL_NAME}")

    # Initialize the database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_name TEXT NOT NULL,
            results TEXT NOT NULL,
            annotated_image TEXT NOT NULL DEFAULT ""
        )
    ''')
    conn.commit()

    # Ensure 'annotated_image' column exists
    add_column_if_not_exists(conn, 'logs', 'annotated_image', 'TEXT NOT NULL DEFAULT ""')
    conn.close()

# GET Home Page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    ''' Returns the homepage '''
    return templates.TemplateResponse('home.html', {"request": request})

# GET Upload Page
@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    ''' Returns the upload form page '''
    return templates.TemplateResponse('upload.html', {"request": request})

# GET Logs Page with Pagination
@app.get("/logs", response_class=HTMLResponse)
def view_logs(request: Request, page: int = 1, per_page: int = 10):
    ''' Returns the logs page with pagination '''
    per_page = min(per_page, 100)  # Prevent excessive per_page values
    offset = (page - 1) * per_page

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM logs')
    total_logs = cursor.fetchone()[0]
    total_pages = (total_logs + per_page - 1) // per_page

    cursor.execute('SELECT * FROM logs ORDER BY id DESC LIMIT ? OFFSET ?', (per_page, offset))
    logs = cursor.fetchall()
    conn.close()

    # Parse results JSON
    parsed_logs = []
    for log in logs:
        parsed_logs.append({
            "id": log["id"],
            "timestamp": log["timestamp"],
            "image_name": log["image_name"],
            "results": json.loads(log["results"]),
            "annotated_image": log["annotated_image"]
        })

    return templates.TemplateResponse('logs.html', {
        "request": request,
        "logs": parsed_logs,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    })

# POST Endpoint to Delete a Specific Log
@app.post("/logs/delete/{log_id}", response_class=RedirectResponse)
def delete_log(log_id: int, request: Request):
    ''' Deletes a specific log entry '''
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM logs WHERE id = ?', (log_id,))
    log = cursor.fetchone()
    if not log:
        conn.close()
        raise HTTPException(status_code=404, detail="Log not found")

    cursor.execute('DELETE FROM logs WHERE id = ?', (log_id,))
    conn.commit()
    conn.close()

    # Redirect back to the logs page, preserving current pagination
    query_params = dict(request.query_params)
    redirect_url = "/logs?" + urlencode(query_params)
    return RedirectResponse(url=redirect_url, status_code=303)

# POST Endpoint to Delete All Logs
@app.post("/logs/delete_all", response_class=RedirectResponse)
def delete_all_logs(request: Request, confirm: Optional[str] = Form(None)):
    ''' Deletes all log entries '''
    if confirm != "DELETE":
        # Prevent accidental deletions
        raise HTTPException(status_code=400, detail="Confirmation required to delete all logs.")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM logs')
    conn.commit()
    conn.close()

    # Redirect back to the logs page
    return RedirectResponse(url="/logs", status_code=303)

# POST Detection Request
@app.post("/", response_class=HTMLResponse)
async def detect_with_server_side_rendering(request: Request,
                            file_list: List[UploadFile] = File(...),
                            img_size: int = Form(640)):
    '''
    Handles image upload, runs detection, and returns results page
    '''
    img_batch = []
    img_str_list = []
    json_results = []
    image_names = []
    annotated_images = []

    for file in file_list:
        contents = await file.read()
        img_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img_np is None:
            continue
        img_batch.append(img_np)
        image_names.append(file.filename)

    if not img_batch:
        return templates.TemplateResponse('upload.html', {
            "request": request,
            "error": "No valid images uploaded."
        })

    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]

    # Run inference
    results = model(img_batch_rgb, size=img_size)

    # Process results
    json_results = results_to_json(results, model)

    # Annotate images and encode to base64
    for img, bbox_list in zip(img_batch, json_results):
        for bbox in bbox_list:
            label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
            plot_one_box(bbox['bbox'], img, label=label,
                        color=colors[int(bbox['class']) % len(colors)], line_thickness=3)
        annotated_image_b64 = base64EncodeImage(img)
        annotated_images.append(annotated_image_b64)
        img_str_list.append(annotated_image_b64)

    # Convert json_results to string for JavaScript
    encoded_json_results = json.dumps(json_results)

    # Log the detection
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cursor = conn.cursor()
    for name, result, annotated_img in zip(image_names, json_results, annotated_images):
        cursor.execute('INSERT INTO logs (timestamp, image_name, results, annotated_image) VALUES (?, ?, ?, ?)',
                       (timestamp, name, json.dumps(result), annotated_img))
    conn.commit()
    conn.close()

    return templates.TemplateResponse('show_results.html', {
        'request': request,
        'bbox_image_data_zipped': zip(img_str_list, json_results),
        'bbox_data_str': encoded_json_results,
    })

@app.post("/detect", response_class=JSONResponse)
async def detect_via_api(request: Request,
                         file_list: List[UploadFile] = File(...),
                         img_size: Optional[int] = Form(640)):
    '''
    API endpoint to perform detection and return JSON results
    '''
    img_batch = []
    json_results = []
    image_names = []
    annotated_images = []

    for file in file_list:
        contents = await file.read()
        img_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img_np is None:
            continue
        img_batch.append(img_np)
        image_names.append(file.filename)

    if not img_batch:
        return JSONResponse(content={"error": "No valid images uploaded."}, status_code=400)

    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]

    # Run inference
    results = model(img_batch_rgb, size=img_size)

    # Process results
    json_results = results_to_json(results, model)

    # Annotate images and encode to base64
    for img, bbox_list in zip(img_batch, json_results):
        for bbox in bbox_list:
            label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
            plot_one_box(bbox['bbox'], img, label=label,
                         color=colors[int(bbox['class']) % len(colors)], line_thickness=3)
        annotated_image_b64 = base64EncodeImage(img)
        annotated_images.append(annotated_image_b64)

    # Log the detection
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cursor = conn.cursor()
    for name, result, annotated_img in zip(image_names, json_results, annotated_images):
        cursor.execute('INSERT INTO logs (timestamp, image_name, results, annotated_image) VALUES (?, ?, ?, ?)',
                       (timestamp, name, json.dumps(result), annotated_img))
    conn.commit()
    conn.close()

    return JSONResponse(content=json_results)


def results_to_json(results, model):
    ''' Converts YOLO model output to JSON (list of list of dicts) '''
    return [
        [
            {
                "class": int(pred[5]),
                "class_name": model.names[int(pred[5])],
                "bbox": [int(x) for x in pred[:4].tolist()],  # [x1, y1, x2, y2]
                "confidence": float(pred[4]),
            }
            for pred in result
        ]
        for result in results.xyxy
    ]

def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    '''
    Plots one bounding box on the image
    '''
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_one_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def base64EncodeImage(img):
    ''' Encodes image to base64 string '''
    _, im_arr = cv2.imencode('.jpg', img)
    im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')
    return im_b64

if __name__ == '__main__':
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', default=8000, type=int)
    parser.add_argument('--precache-models', action='store_true',
                        help='Pre-cache all models in memory upon initialization, otherwise dynamically caches models')
    opt = parser.parse_args()

    # Pre-cache models if needed (currently not used since we have a single default model)
    if opt.precache_models:
        # Placeholder if multiple models are reintroduced in the future
        pass

    # Run Uvicorn with import string to enable reload
    uvicorn.run("server:app", host=opt.host, port=opt.port, reload=True)
