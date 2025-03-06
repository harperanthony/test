import socket
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import argparse  # Added for CLI parsing

# Add argument parsing
parser = argparse.ArgumentParser(description='4K Video Processing Server')
parser.add_argument('--port', type=int, required=True,
                    help='Port number to listen on')
args = parser.parse_args()

HOST = '0.0.0.0'
PORT = args.port
BUFFER_SIZE = 4 * 1024 * 1024  # 4MB buffer
MAX_WORKERS = 4
FRAME_SIZE = (3840, 2160)  # 4K resolution

def process_frame(frame_data):
    # Decode with 4K parameters
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Your processing (grayscale conversion)
    processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Encode with high-quality parameters
    _, encoded = cv2.imencode('.jpg', processed, [
        cv2.IMWRITE_JPEG_QUALITY, 95,
        cv2.IMWRITE_JPEG_SAMPLING_FACTOR, '4:4:4'
    ])
    return encoded.tobytes()

def handle_client(conn):
    with conn:
        while True:
            try:
                # Receive frame size header
                size_header = conn.recv(16)
                if not size_header: break
                frame_size = int.from_bytes(size_header, 'big')
                
                # Receive frame data
                frame_data = bytearray()
                while len(frame_data) < frame_size:
                    chunk = conn.recv(min(BUFFER_SIZE, frame_size - len(frame_data)))
                    if not chunk: break
                    frame_data.extend(chunk)
                
                # Process in parallel
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    processed_data = executor.submit(process_frame, frame_data).result()
                
                # Send processed frame
                conn.sendall(len(processed_data).to_bytes(16, 'big'))
                conn.sendall(processed_data)

            except Exception as e:
                print(f"Error: {e}")
                break

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()
    print(f"4K Server listening on {HOST}:{PORT}")
    
    while True:
        conn, addr = s.accept()
        print(f"New connection from {addr}")
        ThreadPoolExecutor(max_workers=1).submit(handle_client, conn)