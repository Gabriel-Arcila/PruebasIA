import os
import openai
import base64
import pymupdf
from PIL import Image
import base64


endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"
model_embedding = "text-embedding-3-small"
token = os.environ["GITHUB_TOKEN"]
MODEL_NAME = os.getenv("GITHUB_MODEL", model)

client = openai.OpenAI(
    base_url= endpoint,
    api_key= token
)

def open_image_as_base64(filename):
    with open(filename, "rb") as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


filename = "Pdf/plants.pdf"
doc = pymupdf.open(filename)
for i in range(doc.page_count):
    #?doc = pymupdf.open(filename)
    page = doc.load_page(i)
    pix = page.get_pixmap()
    original_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    original_img.save(f"img/page_{i}.png")


user_content = [{"text": "cuantas plantas hay en estas paginas y nombrame cada una?", "type": "text"}]
# Process just the first few pages, as processing all doc.page_count pages is slow
for i in range(5):
    user_content.append({"image_url": {"url": open_image_as_base64(f"img/page_{i}.png")}, "type": "image_url"})

response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": user_content}])

print(response.choices[0].message.content)

