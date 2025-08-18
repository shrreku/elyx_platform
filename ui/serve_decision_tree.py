#!/usr/bin/env python3
"""
Simple HTTP server to serve the Decision Tree Visualizer web app.
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler with CORS support and proper MIME types."""
    
    def end_headers(self):
        # Add CORS headers for web app functionality
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
        # Add cache control headers for JSON files to prevent stale data
        if self.path.endswith('.json'):
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
        
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight OPTIONS request for CORS."""
        self.send_response(200)
        self.end_headers()
    
    def guess_type(self, path):
        """Override MIME type guessing for better web app support."""
        if path.endswith('.js'):
            return 'application/javascript'
        elif path.endswith('.css'):
            return 'text/css'
        elif path.endswith('.html'):
            return 'text/html'
        elif path.endswith('.json'):
            return 'application/json'
        return super().guess_type(path)

def main():
    """Start the HTTP server."""
    ui_dir = Path(__file__).parent
    # Serve from project root so both UI and data directories are reachable
    os.chdir(project_root)
    
    # Server configuration
    PORT = 8080
    Handler = CustomHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print("=" * 60)
            print("🌳 Elyx Platform Decision Tree Visualizer Server")
            print("=" * 60)
            print(f"📁 Serving from project root: {project_root.absolute()}")
            try:
                rel_ui = ui_dir.relative_to(project_root)
            except Exception:
                rel_ui = ui_dir
            print(f"📁 UI path: {rel_ui}")
            print(f"🌐 Server running at: http://localhost:{PORT}")
            print(f"📊 Decision Tree UI: http://localhost:{PORT}/{rel_ui}/decision_tree_visualizer.html")
            print(f"📁 Data files: http://localhost:{PORT}/data/conversation_history.json")
            print("=" * 60)
            print("💡 Open your browser and navigate to the Decision Tree UI URL above")
            print("💡 The web app will automatically load demo data for testing")
            print("💡 Use 'JSON File' option to load your actual conversation data")
            print("=" * 60)
            print("🛑 Press Ctrl+C to stop the server")
            print("=" * 60)
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except OSError as e:
        if getattr(e, 'errno', None) == 48:  # Address already in use (macOS)
            print(f"❌ Port {PORT} is already in use. Try a different port or stop the existing server.")
        else:
            print(f"❌ Server error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
