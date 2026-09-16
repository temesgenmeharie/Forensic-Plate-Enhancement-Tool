"""
Simple launcher for the web-based GUI.
Opens browser automatically.
"""

import os
import sys
import webbrowser
import time
from app.gui.web_app import app

if __name__ == '__main__':
    print("\n" + "="*70)
    print("FORENSIC PLATE ENHANCER - WEB GUI")
    print("="*70)
    print("\n🌐 Starting web server...")
    print("⏳ Please wait...\n")
    
    # Start server in thread
    import threading
    server_thread = threading.Thread(
        target=lambda: app.run(debug=False, host='0.0.0.0', port=8080),
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Open browser
    url = 'http://localhost:8080'
    print(f"✓ Server started on {url}")
    print("\n📱 Opening browser...\n")
    webbrowser.open(url)
    
    print("="*70)
    print("Advanced Deblurring Features Available:")
    print("  ✓ 9 deblurring methods")
    print("  ✓ Real-time visibility metrics")
    print("  ✓ Before/after comparison")
    print("  ✓ JSON report export")
    print("="*70)
    print("\n⚠️  EXPERIMENTAL - Manual verification required\n")
    
    # Keep server running
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)
