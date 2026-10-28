"""
Cruise Passenger Survival Prediction Server
A multi-threaded Python HTTP server providing a REST API and serving
the modern responsive web frontend.
"""

import os
import sys
import json
import mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from model import engine

# Initialize machine learning model engine at startup
print("[Cruise App] Initializing Machine Learning Engine...")
engine.train_or_load()
print(f"[Cruise App] Engine ready! (LR CV: {engine.lr_cv_score}%, RF CV: {engine.rf_cv_score}%)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')

class CruiseHTTPRequestHandler(BaseHTTPRequestHandler):
    server_version = "CruisePredictorServer/2.0"

    def _set_headers(self, status_code=200, content_type='application/json'):
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200, 'text/plain')
        self.wfile.write(b'OK')

    def do_HEAD(self):
        self.do_GET(head_only=True)

    def do_GET(self, head_only=False):
        parsed = urlparse(self.path)
        path = parsed.path

        # API: Fleet Statistics
        if path == '/api/stats':
            self._set_headers(200, 'application/json; charset=utf-8')
            if not head_only:
                self.wfile.write(json.dumps(engine.stats).encode('utf-8'))
            return

        # API: Presets
        if path == '/api/presets':
            self._set_headers(200, 'application/json; charset=utf-8')
            if not head_only:
                self.wfile.write(json.dumps(engine.get_presets()).encode('utf-8'))
            return

        # API: Health Check
        if path == '/api/health':
            self._set_headers(200, 'application/json; charset=utf-8')
            payload = {
                'status': 'healthy',
                'service': 'Cruise Passenger Survival Predictor',
                'models': {
                    'logistic_regression_cv': f"{engine.lr_cv_score}%",
                    'random_forest_cv': f"{engine.rf_cv_score}%"
                }
            }
            if not head_only:
                self.wfile.write(json.dumps(payload).encode('utf-8'))
            return

        # Serve Homepage
        if path in ['/', '/index.html']:
            index_file = os.path.join(BASE_DIR, 'index.html')
            if os.path.exists(index_file):
                self._set_headers(200, 'text/html; charset=utf-8')
                if not head_only:
                    with open(index_file, 'rb') as f:
                        self.wfile.write(f.read())
            else:
                self._set_headers(404, 'text/plain')
                if not head_only:
                    self.wfile.write(b"index.html not found.")
            return

        # Serve Static Files
        if path.startswith('/static/'):
            rel_path = path[len('/static/'):]
            # Prevent directory traversal
            clean_path = os.path.normpath(rel_path)
            if clean_path.startswith('..'):
                self._set_headers(403, 'text/plain')
                if not head_only:
                    self.wfile.write(b"Forbidden")
                return

            file_path = os.path.join(STATIC_DIR, clean_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                ctype, _ = mimetypes.guess_type(file_path)
                if not ctype:
                    ctype = 'application/octet-stream'
                if ctype.startswith('text/') or ctype in ['application/javascript', 'application/json']:
                    ctype += '; charset=utf-8'

                self._set_headers(200, ctype)
                if not head_only:
                    with open(file_path, 'rb') as f:
                        self.wfile.write(f.read())
            else:
                self._set_headers(404, 'text/plain')
                if not head_only:
                    self.wfile.write(b"Static file not found.")
            return

        # Fallback 404
        self._set_headers(404, 'text/plain')
        if not head_only:
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length <= 0:
                self._set_headers(400, 'application/json')
                self.wfile.write(json.dumps({'error': 'No input data provided'}).encode('utf-8'))
                return

            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
                prediction_result = engine.predict_passenger(data)
                self._set_headers(200, 'application/json; charset=utf-8')
                self.wfile.write(json.dumps(prediction_result).encode('utf-8'))
            except Exception as e:
                self._set_headers(500, 'application/json')
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            return

        self._set_headers(404, 'text/plain')
        self.wfile.write(b"404 Not Found")

    def log_message(self, format, *args):
        # Clean server logging format
        sys.stderr.write(f"[CruiseServer] {self.address_string()} - {format % args}\n")

def run_server(port=8000):
    server_address = ('0.0.0.0', port)
    try:
        httpd = ThreadingHTTPServer(server_address, CruiseHTTPRequestHandler)
        print(f"\n========================================================")
        print(f"  CRUISE PASSENGER PREDICTION WEB APP RUNNING")
        print(f"  Access URL: http://localhost:{port}")
        print(f"  Network URL: http://127.0.0.1:{port}")
        print(f"========================================================\n")
        httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Port {port} in use, trying {port + 1}...")
            run_server(port + 1)
        else:
            raise e

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
