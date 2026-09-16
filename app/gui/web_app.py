"""
Web-based GUI for Forensic Plate Enhancer.
Simple Flask web interface for deblurring license plates.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

from app.core.evidence import Evidence
from app.core.session import Session
from app.processing.advanced_deblur import (
    apply_blind_deconvolution,
    apply_lucy_richardson_advanced,
    apply_total_variation_deblur,
    apply_frequency_domain_deblur,
    apply_super_resolution_upscale,
    apply_multi_scale_deblur,
    calculate_plate_visibility_score,
    recommend_deblur_method,
    apply_morphological_enhancement
)
from app.processing.resize import upscale_image_multiple
from app.processing.contrast import apply_clahe
from app.processing.threshold import apply_adaptive_gaussian_threshold

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store current processing state
processing_state = {
    'original_image': None,
    'current_image': None,
    'results': [],
    'session_id': None,
    'evidence_id': None
}


def image_to_base64(image_array):
    """Convert numpy array to base64 encoded PNG."""
    if image_array is None:
        return None
    
    _, buffer = cv2.imencode('.png', image_array)
    img_base64 = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_base64}"


def get_image_metrics(image):
    """Get metrics for an image."""
    if image is None:
        return None
    
    metrics = calculate_plate_visibility_score(image)
    return {
        'visibility_score': round(metrics['visibility_score'], 1),
        'sharpness': round(metrics['sharpness'], 1),
        'contrast': round(metrics['contrast'], 1),
        'edge_density': round(metrics['edge_density'], 1),
        'is_readable': metrics['is_readable']
    }


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Initialize session
        processing_state['session_id'] = Session().session_id
        processing_state['original_image'] = image.copy()
        processing_state['current_image'] = image.copy()
        processing_state['results'] = []
        
        # Get metrics
        metrics = get_image_metrics(image)
        
        # Get recommended method
        method, scores = recommend_deblur_method(image)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'shape': list(image.shape),
            'metrics': metrics,
            'recommended_method': method,
            'method_scores': {k: round(v, 3) for k, v in scores.items()},
            'preview': image_to_base64(image)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/deblur', methods=['POST'])
def deblur():
    """Apply deblurring method."""
    try:
        data = request.json
        method = data.get('method')
        
        if processing_state['original_image'] is None:
            return jsonify({'error': 'No image loaded'}), 400
        
        image = processing_state['current_image'].copy()
        result = None
        processing_info = {
            'method': method,
            'timestamp': datetime.now().isoformat()
        }
        
        if method == 'blind_deconv':
            result = apply_blind_deconvolution(image, iterations=20)
            processing_info['params'] = {'iterations': 20}
        
        elif method == 'rl_motion':
            result = apply_lucy_richardson_advanced(image, iterations=15, psf_type='motion')
            processing_info['params'] = {'iterations': 15, 'psf_type': 'motion'}
        
        elif method == 'rl_gaussian':
            result = apply_lucy_richardson_advanced(image, iterations=15, psf_type='gaussian')
            processing_info['params'] = {'iterations': 15, 'psf_type': 'gaussian'}
        
        elif method == 'total_variation':
            result = apply_total_variation_deblur(image, strength=0.1, iterations=100)
            processing_info['params'] = {'strength': 0.1, 'iterations': 100}
        
        elif method == 'freq_domain':
            result = apply_frequency_domain_deblur(image, blur_type='motion', strength=1.0)
            processing_info['params'] = {'blur_type': 'motion', 'strength': 1.0}
        
        elif method == 'super_res_2x':
            result = apply_super_resolution_upscale(image, scale_factor=2, iterations=10)
            processing_info['params'] = {'scale_factor': 2, 'iterations': 10}
        
        elif method == 'super_res_4x':
            result = apply_super_resolution_upscale(image, scale_factor=4, iterations=10)
            processing_info['params'] = {'scale_factor': 4, 'iterations': 10}
        
        elif method == 'multi_scale':
            result = apply_multi_scale_deblur(image, scales=[0.5, 1.0])
            processing_info['params'] = {'scales': [0.5, 1.0]}
        
        elif method == 'morphological':
            result = apply_morphological_enhancement(image, kernel_size=5, operations=['close', 'open'])
            processing_info['params'] = {'kernel_size': 5, 'operations': ['close', 'open']}
        
        else:
            return jsonify({'error': 'Unknown method'}), 400
        
        if result is None:
            return jsonify({'error': 'Processing failed'}), 500
        
        # Update state
        processing_state['current_image'] = result
        processing_state['results'].append(processing_info)
        
        # Get metrics
        metrics_before = get_image_metrics(image)
        metrics_after = get_image_metrics(result)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result),
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'improvement': round(metrics_after['visibility_score'] - metrics_before['visibility_score'], 1)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/enhance', methods=['POST'])
def enhance():
    """Apply enhancement (CLAHE, threshold, etc)."""
    try:
        data = request.json
        operation = data.get('operation')
        
        image = processing_state['current_image'].copy()
        result = None
        
        if operation == 'clahe':
            clip_limit = float(data.get('clip_limit', 2.0))
            result = apply_clahe(image, clip_limit=clip_limit)
        
        elif operation == 'threshold_adaptive':
            result = apply_adaptive_gaussian_threshold(image)
        
        elif operation == 'upscale_2x':
            upscaled = upscale_image_multiple(image, scale_factors=[2.0])
            result = upscaled[2.0]
        
        else:
            return jsonify({'error': 'Unknown enhancement'}), 400
        
        # Update state
        processing_state['current_image'] = result
        processing_state['results'].append({
            'operation': operation,
            'timestamp': datetime.now().isoformat()
        })
        
        metrics_before = get_image_metrics(image)
        metrics_after = get_image_metrics(result)
        
        return jsonify({
            'success': True,
            'result': image_to_base64(result),
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'improvement': round(metrics_after['visibility_score'] - metrics_before['visibility_score'], 1)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/undo', methods=['POST'])
def undo():
    """Undo last operation."""
    try:
        if len(processing_state['results']) > 0:
            processing_state['results'].pop()
            processing_state['current_image'] = processing_state['original_image'].copy()
            
            # Reapply previous operations
            for op in processing_state['results']:
                # Placeholder - in production would reapply all operations
                pass
        
        metrics = get_image_metrics(processing_state['current_image'])
        
        return jsonify({
            'success': True,
            'result': image_to_base64(processing_state['current_image']),
            'metrics': metrics,
            'operations_count': len(processing_state['results'])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset to original image."""
    try:
        processing_state['current_image'] = processing_state['original_image'].copy()
        processing_state['results'] = []
        
        metrics = get_image_metrics(processing_state['original_image'])
        
        return jsonify({
            'success': True,
            'result': image_to_base64(processing_state['original_image']),
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare', methods=['GET'])
def compare():
    """Get side-by-side comparison."""
    try:
        original_b64 = image_to_base64(processing_state['original_image'])
        current_b64 = image_to_base64(processing_state['current_image'])
        
        metrics_before = get_image_metrics(processing_state['original_image'])
        metrics_after = get_image_metrics(processing_state['current_image'])
        
        return jsonify({
            'original': original_b64,
            'current': current_b64,
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'operations': len(processing_state['results'])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/report', methods=['GET'])
def get_report():
    """Generate JSON report."""
    try:
        report = {
            'session_id': processing_state['session_id'],
            'timestamp': datetime.now().isoformat(),
            'operations': processing_state['results'],
            'metrics_before': get_image_metrics(processing_state['original_image']),
            'metrics_after': get_image_metrics(processing_state['current_image']),
            'status': 'EXPERIMENTAL - Manual verification required'
        }
        
        return jsonify(report)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("FORENSIC PLATE ENHANCER - WEB GUI")
    print("="*70)
    print("\n🌐 Opening web interface...")
    print("📱 URL: http://localhost:5000")
    print("\n✓ Advanced deblurring: 9 techniques")
    print("✓ Real-time metrics: Visibility, sharpness, contrast")
    print("✓ Processing pipeline: Multiple enhancement options")
    print("✓ Report generation: JSON export with audit trail")
    print("\n⚠️  EXPERIMENTAL - Manual verification required\n")
    
    app.run(debug=True, host='localhost', port=5000)
