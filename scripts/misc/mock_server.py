from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        
        # Send reasoning
        self.wfile.write(b'data: {"choices":[{"delta":{"reasoning_content":"Thinking..."}}]}\n\n')
        # Send content
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"Hello world"}}]}\n\n')
        self.wfile.write(b'data: [DONE]\n\n')

HTTPServer(('localhost', 8001), Handler).serve_forever()
