from fastapi import FastAPI, File, Form, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
import io
import cv2
import numpy as np
import sys
import base64
import re
# import pytesseract
# import googletrans
# from googletrans import Translator
# from gtts import gTTS

# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\vikas.LAPTOP-RRF59END\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

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

    
ocr_dict = {
    'Afrikaans': 'afr',
    'Albanian': 'sqi',
    'Amharic': 'amh',
    'Arabic': 'ara',
    'Armenian': 'hye',
    'Azerbaijani': 'aze',
    'Basque': 'eus',
    'Belarusian': 'bel',
    'Bengali': 'ben',
    'Bosnian': 'bos',
    'Bulgarian': 'bul',
    'Catalan': 'cat',
    'Cebuano': 'ceb',
    'Chichewa': 'ny',
    'Chinese (Simplified)': 'chi_sim',
    'Chinese (Traditional)': 'chi_tra',
    'Corsican': 'cos',
    'Croatian': 'hrv',
    'Czech': 'ces',
    'Danish': 'dan',
    'Dutch': 'nld',
    'English': 'eng',
    'Esperanto': 'epo',
    'Estonian': 'est',
    'Filipino': 'fil',
    'Finnish': 'fin',
    'French': 'fra',
    'Frisian': 'fry',
    'Galician': 'glg',
    'Georgian': 'kat',
    'German': 'deu',
    'Greek': 'ell',
    'Gujarati': 'guj',
    'Haitian Creole': 'hat',
    'Hausa': 'ha',
    'Hawaiian': 'haw',
    'Hebrew': 'heb',
    'Hindi': 'hin',
    'Hmong': 'hmn',
    'Hungarian': 'hun',
    'Icelandic': 'isl',
    'Igbo': 'ig',
    'Indonesian': 'ind',
    'Irish': 'gle',
    'Italian': 'ita',
    'Japanese': 'jpn',
    'Javanese': 'jav',
    'Kannada': 'kan',
    'Kazakh': 'kaz',
    'Khmer': 'khm',
    'Korean': 'kor',
    'Kurdish (Kurmanji)': 'ku',
    'Kyrgyz': 'kir',
    'Lao': 'lao',
    'Latin': 'lat',
    'Latvian': 'lav',
    'Lithuanian': 'lit',
    'Luxembourgish': 'ltz',
    'Macedonian': 'mkd',
    'Malagasy': 'mlg',
    'Malay': 'msa',
    'Malayalam': 'mal',
    'Maltese': 'mlt',
    'Maori': 'mri',
    'Marathi': 'mar',
    'Mongolian': 'mon',
    'Myanmar (Burmese)': 'mya',
    'Nepali': 'nep',
    'Norwegian': 'nor',
    'Odia': 'ori',
    'Pashto': 'pus',
    'Persian': 'fas',
    'Polish': 'pol',
    'Portuguese': 'por',
    'Punjabi': 'pan',
    'Romanian': 'ron',
    'Russian': 'rus',
    'Samoan': 'sm',
    'Scots Gaelic': 'gla',
    'Serbian': 'srp',
    'Sesotho': 'st',
    'Shona': 'sna',
    'Sindhi': 'snd',
    'Sinhala': 'sin',
    'Slovak': 'slk',
    'Slovenian': 'slv',
    'Somali': 'som',
    'Spanish': 'spa',
    'Sundanese': 'sun',
    'Swahili': 'swa',
    'Swedish': 'swe',
    'Tagalog (Filipino)': 'fil',
    'Tajik': 'tgk',
    'Tamil': 'tam',
    'Telugu': 'tel',
    'Thai': 'tha',
    'Turkish': 'tur',
    'Ukrainian': 'ukr',
    'Urdu': 'urd',
    'Uyghur': 'uig',
    'Uzbek': 'uzb',
    'Vietnamese': 'vie',
    'Welsh': 'cym',
    'Xhosa': 'xho',
    'Yiddish': 'yid',
    'Yoruba': 'yor',
    'Zulu': 'zul'
}

google_trans_dict = {
    'Afrikaans': 'af',
    'Albanian': 'sq',
    'Amharic': 'am',
    'Arabic': 'ar',
    'Armenian': 'hy',
    'Azerbaijani': 'az',
    'Basque': 'eu',
    'Belarusian': 'be',
    'Bengali': 'bn',
    'Bosnian': 'bs',
    'Bulgarian': 'bg',
    'Catalan': 'ca',
    'Cebuano': 'ceb',
    'Corsican': 'co',
    'Croatian': 'hr',
    'Czech': 'cs',
    'Danish': 'da',
    'Dutch': 'nl',
    'English': 'en',
    'Esperanto': 'eo',
    'Estonian': 'et',
    'Filipino': 'tl',
    'Finnish': 'fi',
    'French': 'fr',
    'Frisian': 'fy',
    'Galician': 'gl',
    'Georgian': 'ka',
    'German': 'de',
    'Greek': 'el',
    'Gujarati': 'gu',
    'Hindi': 'hi',
    'Hungarian': 'hu',
    'Icelandic': 'is',
    'Indonesian': 'id',
    'Irish': 'ga',
    'Italian': 'it',
    'Japanese': 'ja',
    'Javanese': 'jw',
    'Kannada': 'kn',
    'Kazakh': 'kk',
    'Khmer': 'km',
    'Korean': 'ko',
    'Kurdish (Kurmanji)': 'ku',
    'Kyrgyz': 'ky',
    'Lao': 'lo',
    'Latin': 'la',
    'Latvian': 'lv',
    'Lithuanian': 'lt',
    'Luxembourgish': 'lb',
    'Macedonian': 'mk',
    'Malagasy': 'mg',
    'Malay': 'ms',
    'Malayalam': 'ml',
    'Maltese': 'mt',
    'Maori': 'mi',
    'Marathi': 'mr',
    'Mongolian': 'mn',
    'Myanmar (Burmese)': 'mya',
    'Nepali': 'ne',
    'Norwegian': 'no',
    'Odia': 'ori',
    'Pashto': 'ps',
    'Persian': 'fa',
    'Polish': 'pl',
    'Portuguese': 'pt',
    'Punjabi': 'pa',
    'Romanian': 'ro',
    'Russian': 'ru',
    'Samoan': 'sm',
    'Scots Gaelic': 'gd',
    'Serbian': 'sr',
    'Sesotho': 'st',
    'Shona': 'sn',
    'Sindhi': 'sd',
    'Sinhala': 'si',
    'Slovak': 'sk',
    'Slovenian': 'sl',
    'Somali': 'so',
    'Spanish': 'es',
    'Sundanese': 'su',
    'Swahili': 'sw',
    'Swedish': 'sv',
    'Tagalog (Filipino)': 'tl',
    'Tajik': 'tg',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Thai': 'th',
    'Turkish': 'tr',
    'Ukrainian': 'uk',
    'Urdu': 'ur',
    'Uyghur': 'ug',
    'Uzbek': 'uz',
    'Vietnamese': 'vi',
    'Welsh': 'cy',
    'Xhosa': 'xh',
    'Yiddish': 'yi',
    'Yoruba': 'yo',
    'Zulu': 'zu'
}


def preprocessing(url):
    try:
        ip_image=cv2.imread(url) 
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


def ocr(url, fromlang):
    try:
        img = cv2.imread(url)
         # Perform OCR using pytesseract
        text = pytesseract.image_to_string(img, lang=fromlang)  # change lang value to "from_lang"

        # Save recognized text to a file
        print(f"text is {text}")
        with open("textfile/img_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        # print(text)   
        return text
        
    except Exception as e:
        print("Error on line {}".format(sys.exc_info()[-1].tb_lineno))
        return



def translateText(fromlang, toLang):
    with open('textfile/img_text.txt', 'r') as file:
    # Read the contents of the file into a variable called text
        text = file.read()
# Print the contents of the text variable
    print("Original Text:", text) 

    extracted_text = text.replace('\n', ' ')
    extracted_text = extracted_text.replace('_', '')
    extracted_text = extracted_text.replace('|', '')
    
    translator=Translator() 
    detected_language = translator.detect(extracted_text).lang
    print("Detected Language:", detected_language)
    res = translator.translate(extracted_text, dest='kn')
    print("\nres.text:\n",res.text)
    
    with open("textfile/translated_text.txt", "w", encoding="utf-8") as f:
        f.write(res.text)
    return res.text




@app.post("/upload")
async def upload_file(file: bytes = File(...), fromLang: str = Form(...), toLang: str = Form(...)):
    try:
        image = Image.open(io.BytesIO(file))
        image.save(f'images/unsharpened_image.jpg')
        preprocessing()
        extracted = ocr(ocr_dict[fromLang])
        extracted = extracted.replace('\n', ' ')
        extracted = extracted.replace('_', '')
        extracted = extracted.replace('|', '')
        res = translateText(google_trans_dict[fromLang], google_trans_dict[toLang])
        return {
            "data": {
                "translated_text": res,
                "original_text": extracted
            },
            "status":200,
            "message": "Message translated successfully"  
        }
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return {
            "data": [],
            "status": 500,
            "message": "An error occurred while processing your request."
        }


@app.post("/camera")
async def camera(file: bytes = File(...), fromLang: str = Form(...), toLang: str = Form(...)):
    try:
        image_data = file.decode('utf-8')[len('data:image/jpeg;base64,'):] 
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        img.save('images/unsharpened_camera_image.jpg')
        preprocessing('images/unsharpened_camera_image.jpg')
        extracted = ocr(ocr_dict[fromLang])
        extracted = extracted.replace('\n', ' ')
        extracted = extracted.replace('_', '')
        extracted = extracted.replace('|', '')
        res = translateText(google_trans_dict[fromLang], google_trans_dict[toLang])

        return {
            "data": {
                "translated_text": res,
                "extracted_text": extracted,
            },
            "message": "Image saved successfully",
            "status": 200
        }
    except Exception as e:
        return {
            "data": [],
            "message": str(e),
            "status": 500
        }
