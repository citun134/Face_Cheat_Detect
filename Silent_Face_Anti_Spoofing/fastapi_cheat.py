import os
import uuid
import time
import aiofiles
import logging
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from image_cheat_detection import FaceDetectors  # Model phát hiện gian lận bằng khuôn mặt

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thêm file log ghi kết quả
LOG_FILE = "detection_results.log"

# Khởi tạo app và model
app = FastAPI(title="Face Detection API", description="API phát hiện gian lận qua khuôn mặt", version="1.0")
facedetector = FaceDetectors()

# Middleware CORS nếu cần mở rộng hệ thống sau này
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thư mục lưu tạm ảnh upload
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Giao diện HTML đơn giản cho test
@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <html>
        <head><title>Upload Image</title></head>
        <body>
            <h2>Upload an image</h2>
            <form action="/detect" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Upload and Detect">
            </form>
        </body>
    </html>
    """

# API phát hiện gian lận/khuôn mặt từ ảnh
@app.post("/detect", summary="Phát hiện gian lận qua ảnh", description="Nhận ảnh và phát hiện khuôn mặt nghi ngờ")
async def detect_face(request: Request, file: UploadFile = File(...)):
    # Kiểm tra định dạng ảnh
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ ảnh JPEG hoặc PNG.")

    # Tạo tên file tạm duy nhất
    file_ext = os.path.splitext(file.filename)[1]
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        # Ghi file async
        async with aiofiles.open(temp_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        # Gọi model
        start_time = time.time()
        result = facedetector.face_detector(temp_path)
        end_time = time.time()

        result["execution_time"] = f"{end_time - start_time:.2f} seconds"

        # Ghi log xử lý
        log_message = f"File '{file.filename}' xử lý trong {end_time - start_time:.2f}s | Kết quả: {result}\n"
        logger.info(log_message)
        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(log_message)
        # file_handler = logging.FileHandler("detection_results.log", encoding="utf-8")
        # file_handler.setLevel(logging.INFO)
        # logger.addHandler(file_handler)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Lỗi xử lý file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi trong quá trình xử lý ảnh")

    finally:
        # Xoá file tạm
        if os.path.exists(temp_path):
            os.remove(temp_path)

# async def detect_face(file: UploadFile = File(...)):
#     # Đọc nội dung ảnh từ UploadFile thành bytes
#     image_bytes = await file.read()
#
#     # Chuyển bytes thành mảng numpy để xử lý bằng OpenCV
#     np_array = np.frombuffer(image_bytes, np.uint8)
#     img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # img là ảnh dạng BGR
#
#     # Gọi hàm phát hiện khuôn mặt (giả sử nó nhận ảnh numpy)
#     start_time = time.time()
#     result = facedetector.face_detector(img)  # Gọi hàm face_detector với ảnh numpy
#     end_time = time.time()
#
#     result["execution_time"] = f"{end_time - start_time:.2f} seconds"
#     return JSONResponse(content=result)
