from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.requests import Request
import io

from torch import load, device
import torchvision.transforms as transforms
from models import Encoder, Decoder
from utils import get_caption
from data import Vocabulary

cfg = {'lr': 0.0005577729577198625,
    'dropout': 0.31784547520801426,
    'momentum': 0.014810723778791727,
    'n_layers': 1,
    'batch_size': 32,
    'embed_size': 128,
    'hidden_size': 512,
    'num_epochs': 10,
    'device': "cpu"}
vocab = Vocabulary(freq_threshold=5)

def load_model():
    model_path = "./data/model-6.ckpt"
    checkpoint = load(model_path, map_location=device('cpu'))
    encoder = Encoder(cfg["embed_size"], cfg["momentum"]).to(cfg["device"])
    decoder = Decoder(cfg["embed_size"], 
                    cfg["hidden_size"], 
                    len(vocab),
                    cfg["n_layers"],
                    cfg["dropout"]).to(cfg["device"])

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    encoder.eval()
    decoder.eval()

    return encoder, decoder

def transform_image(image):
    image_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(), 
                    transforms.Normalize((0.485, 0.456, 0.406), 
                                        (0.229, 0.224, 0.225))])
    image = image_transform(image)
    image = image.to(cfg["device"])
    image = image.unsqueeze(0)
    return image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:8080/"],
    allow_credentials=True,
    allow_methods=["DELETE", "GET", "POST", "PUT"],
    allow_headers=["*"])

@app.get("/healthcheck")
async def healthcheck():
    msg = (
        "this sentence is already halfway over, "
        "and still hasn't said anything at all"
    )
    return {"message": msg}

@app.post("/predict")
async def predict(request: Request, 
            file: bytes = File(...)):
    data = {"success": False}

    encoder, decoder = load_model()

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        image_bytes = io.BytesIO(file)
        image = Image.open(image_bytes).convert("RGB")
        image = transform_image(image)

        candidates_beam = get_caption(image, 
                                encoder, 
                                decoder, 
                                vocab,
                                "beam",
                                True)

        data["candidates"] = candidates_beam
        data["success"] = True
    return data