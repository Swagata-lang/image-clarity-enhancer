
import os
import sys
import warnings
import torch
import cv2
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename


from flask_cors import CORS

# ========================
# 1. Configuration
# ========================
app = Flask(__name__)
CORS(app, origins=["https://image-enhancer-6d278d.netlify.app"])

application = app 

# Settings
app.config.update({
    'UPLOAD_FOLDER': 'static/uploads',
    'OUTPUT_FOLDER': 'static/outputs',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg'},
    'MODEL_PATH': 'experiments/pretrained_models/ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'
})

# ========================
# 2. BasicSR Integration
# ========================
# Suppress registry warnings
warnings.filterwarnings("ignore", message=".*was already registered in.*")

# Add BasicSR to path
BASICSR_PATH = os.path.join(os.path.dirname(__file__), "basicsr")
sys.path.insert(0, BASICSR_PATH)

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    # Fallback for direct file import
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rrdbnet_arch", 
        os.path.join(BASICSR_PATH, "basicsr", "archs", "rrdbnet_arch.py")
    )
    rrdbnet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rrdbnet)
    RRDBNet = rrdbnet.RRDBNet

# ========================
# 3. AI Model Setup
# ========================
class ImageEnhancer:
    def __init__(self):
        print("DEBUG",BASICSR_PATH)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        self.model.load_state_dict(torch.load((app.config['MODEL_PATH']))['params'], strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)
    

    #  self.model = RRDBNet(num_in_ch=3, num_out_ch=3)
        # self.model.load_state_dict(torch.load(app.config['MODEL_PATH']))
        # self.model.to(self.device).eval()
    
    def enhance(self, img_path):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)/255.
            with torch.no_grad():
                output = self.model(img.to(self.device))
            return output.squeeze().permute(1,2,0).clamp(0,1).cpu().numpy()*255
        except Exception as e:
            print(f"Enhancement failed: {str(e)}")
            return None

enhancer = ImageEnhancer()

# ========================
# 4. Flask Routes
# ========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-form')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save original
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Process image
        output = enhancer.enhance(input_path)
        if output is None:
            return jsonify({'error': 'AI processing failed'}), 500
        
        # Save result
        output_filename = f"enhanced_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        cv2.imwrite(output_path, output)
        
        return jsonify({
            'original': f'/static/uploads/{filename}',
            'enhanced': f'/static/outputs/{output_filename}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ========================
# 5. Startup Checks
# ========================
def check_dependencies():
    required = [
        ('BasicSR', os.path.exists(BASICSR_PATH)),
        ('Model', os.path.exists(app.config['MODEL_PATH'])),
        ('CUDA', torch.cuda.is_available())
    ]
    
    print("\n=== SYSTEM CHECK ===")
    for name, status in required:
        print(f"{'✅' if status else '❌'} {name}")
    print("===================")

if __name__ == '__main__':
    # Create folders if missing
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    # Verify setup
    check_dependencies()
    
    # Start server
    port = int(os.environ.get("PORT", 8000))  # Railway provides $PORT
    app.run(host='0.0.0.0', port=port)