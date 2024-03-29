from fastapi import FastAPI, File, UploadFile, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import io
import cv2
import numpy as np
import sys
import pytesseract

app = FastAPI()
app.add_middleware( 
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
)   

class RequestBody(BaseModel):
    file: bytes = File(...)
    fromLang: str
    toLang: str

    # pre-processing fuunction
def preprocessing():
    try:
        ip_image=cv2.imread('images/unsharpened_image.jpg') 
        print(ip_image.shape)
        
        # cv2.imshow("frame received",ip_image)
        cv2.waitKey(0)

        # # Create the sharpening kernel 
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 

        # # Sharpen the image 
        sharpened_image = cv2.filter2D(ip_image, -1, kernel) 
        sharpened_image = cv2.filter2D(sharpened_image, -1, kernel) 
        # #Save the image 
        cv2.imwrite('images/sharpened_image.jpg', sharpened_image)
        return
    except Exception as e:
        print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))
        return
    
def ocr():
    img = cv2.imread(r'images/sharpened_image.jpg')
    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(img, lang='eng')  # change lang value to "from_lang"

# Save recognized text to a file
    with open("textfile/img_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return


@app.post("/upload")
async def upload_file(file: bytes = File(...), fromLang: str = Form(...), toLang: str = Form(...)):
    image = Image.open(io.BytesIO(file))
    # print(fromLang, toLang)
    
    image.save(f'images/unsharpened_image.jpg')
    # pre-processing fuunction
    preprocessing()
    ocr()
    return


# @app.post("/upload")
# async def upload_file(request: RequestBody):
#     print(request["fromLang"], request["toLang"])
#     return
#     image = Image.open(io.BytesIO(file.file.read()))
#     # with open(f'images/{file.filename}', 'wb') as f:
#     #     f.write(file.file.read())
#     # image.save(f'images/{file.filename}')

#     # return {"filename": file.filename, "fromLang": fromLang, "toLang": toLang}