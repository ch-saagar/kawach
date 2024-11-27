from flask import Flask, request, send_file, jsonify, render_template
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import io
import os

from sentence_transformers import SentenceTransformer, util
import numpy as np


app = Flask(__name__)

# Paths for temporary files
TEXT_PDF_PATH = "text_only.pdf"
IMAGES_PDF_PATH = "images_only.pdf"

# Route to serve the main HTML page
@app.route('/')
def home():
    return render_template('index.html')  # This assumes 'index.html' is in a folder named 'templates'

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global pdf_file
    pdf_file = request.files.get('pdf')
    if not pdf_file:
        return jsonify({"error": "No file uploaded"}), 400
    
    pdf_path = "uploaded.pdf"
    pdf_file.save(pdf_path)
    
    return jsonify({"message": "PDF uploaded successfully"}), 200

@app.route('/extract_text', methods=['GET'])
def extract_text():
    with fitz.open("uploaded.pdf") as pdf_document:
        c = canvas.Canvas(TEXT_PDF_PATH, pagesize=letter)
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            page_text = page.get_text("text")
            c.drawString(72, 750, f"Page {page_num + 1} Text:")
            y = 730
            for line in page_text.splitlines():
                c.drawString(72, y, line)
                y -= 12
                if y < 72:
                    c.showPage()
                    y = 750
            c.showPage()
        c.save()
    
    return send_file(TEXT_PDF_PATH, as_attachment=True)


@app.route('/extract_images', methods=['GET'])
def extract_images():
    try:
        with fitz.open("uploaded.pdf") as pdf_document:
            images_pdf = fitz.open()
            image_found = False  # Flag to check if any images are found

            # Loop through each page and extract images
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)

                # If images are found, process them
                if image_list:
                    image_found = True
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        img_bytes = base_image["image"]
                        image = Image.open(io.BytesIO(img_bytes))
                        img_page = images_pdf.new_page(width=image.width, height=image.height)
                        img_page.insert_image(fitz.Rect(0, 0, image.width, image.height), stream=img_bytes)

            # If no images were found, return an error
            if not image_found:
                return jsonify({"error": "No images found in the PDF to extract."}), 400

            # Save the images PDF if images were extracted
            images_pdf.save(IMAGES_PDF_PATH)
            images_pdf.close()

        # Return the images-only PDF as a downloadable file
        return send_file(IMAGES_PDF_PATH, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/extract_both', methods=['GET'])
def extract_both():
    # Generate text and image PDFs
    extract_text()
    extract_images()
    
    # Return URLs for both files
    return jsonify({
        "text_url": request.url_root + 'download_text_pdf',
        "images_url": request.url_root + 'download_images_pdf'
    })

# Separate routes for downloading the generated text and image PDFs
@app.route('/download_text_pdf', methods=['GET'])
def download_text_pdf():
    return send_file(TEXT_PDF_PATH, as_attachment=True)

@app.route('/download_images_pdf', methods=['GET'])
def download_images_pdf():
    return send_file(IMAGES_PDF_PATH, as_attachment=True)







# Load the model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Store text segments and embeddings
text_segments = []
embeddings = []

def preprocess_pdf():
    global text_segments, embeddings, page_images
    text_segments = []
    embeddings = []
    page_images = {}

    with fitz.open("uploaded.pdf") as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Extract text and associate it with the page number
            page_text = page.get_text("text")
            text_segments.extend([(page_num, segment) for segment in page_text.split("\n\n")])
            
            # Extract images and store them by page number
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                img_bytes = base_image["image"]
                if page_num not in page_images:
                    page_images[page_num] = []
                page_images[page_num].append(img_bytes)

        # Generate embeddings for the text segments
        embeddings = model.encode([segment for _, segment in text_segments])

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Encode the question and calculate similarity with text segments
    question_embedding = model.encode(question)
    similarities = util.cos_sim(question_embedding, embeddings)
    best_match_index = np.argmax(similarities)
    page_num, answer = text_segments[best_match_index]

    # Retrieve images for the corresponding page
    related_images = page_images.get(page_num, [])
    image_files = []
    
    # Save images temporarily for returning
    for idx, img_bytes in enumerate(related_images):
        img_path = f"temp_image_{page_num}_{idx}.png"
        with open(img_path, "wb") as img_file:
            img_file.write(img_bytes)
        image_files.append(img_path)

    # Return the answer and image URLs
    response = {
        "answer": answer,
        "images": [request.url_root + f"get_image?filename={os.path.basename(img)}" for img in image_files]
    }
    return jsonify(response)

@app.route('/get_image', methods=['GET'])
def get_image():
    filename = request.args.get('filename')
    return send_file(filename, as_attachment=True)

# Preprocess PDF on start
if ("uploaded.pdf"):
    preprocess_pdf()

if __name__ == '__main__':
    app.run(debug=True)












