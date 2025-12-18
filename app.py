from fastapi import FastAPI, UploadFile, File
import os, shutil

app = FastAPI()

DB_PATH = "face_db"
UPLOAD_DIR = "uploads"
TEST_IMAGE_PATH = os.path.join(UPLOAD_DIR, "test.jpeg")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload-test-image")
async def upload_test_image(file: UploadFile = File(...)):
    with open(TEST_IMAGE_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "uploaded"}


@app.get("/verify-face")
def verify_face():
    from deepface import DeepFace   # 👈 IMPORT HERE (NOT TOP)

    best_match = None
    best_distance = 999

    for person in os.listdir(DB_PATH):
        for img in os.listdir(os.path.join(DB_PATH, person)):
            img_path = os.path.join(DB_PATH, person, img)
            try:
                result = DeepFace.verify(
                    img1_path=TEST_IMAGE_PATH,
                    img2_path=img_path,
                    model_name="Facenet",
                    detector_backend="opencv",  # avoids retinaface
                    enforce_detection=True
                )
                if result["verified"] and result["distance"] < best_distance:
                    best_match = person
                    best_distance = result["distance"]
            except:
                pass

    return {"matched": best_match, "distance": best_distance}
