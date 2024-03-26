from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import io


app = FastAPI()
app.add_middleware( 
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"], 
)   

# class RequestBody(BaseModel):
#     fromLang: str
#     toLang: str
#     file: UploadFile = File(...)


@app.post("/upload")
async def upload_file(file: bytes = File(...)):
    image = Image.open(io.BytesIO(file))
    image.show()

    return


# @app.post("/upload")
# async def upload_file(request: RequestBody):
#     print(request["fromLang"], request["toLang"])
#     return
    # image = Image.open(io.BytesIO(file.file.read()))
    # with open(f'images/{file.filename}', 'wb') as f:
    #     f.write(file.file.read())
    # image.save(f'images/{file.filename}')

    # return {"filename": file.filename, "fromLang": fromLang, "toLang": toLang}