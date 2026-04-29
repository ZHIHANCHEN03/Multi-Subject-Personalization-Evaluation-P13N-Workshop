#!/usr/bin/env python3
import http.server
import socketserver
import os
from pathlib import Path

# Change to the project root directory
os.chdir('/Users/bytedance/Documents/multi_subject_generation')

PORT = 8001

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print(f"Serving at http://localhost:{PORT}")
    print(f"Open http://localhost:{PORT}/Multi-Subject-Personalization-Evaluation-P13N-Workshop/Model_Training/vlm_multisubject_eval/results/visualization.html in your browser")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")