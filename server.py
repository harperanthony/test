import socket
import cv2
import numpy as np
import argparse
from threading import Thread
import time

parser = argparse.ArgumentParser(description='High Speed Video Server')
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()

HOST = '0.0.0.0'
PORT = args.port
BUFFER_SIZE = 4 * 1024 * 1024  # 4MB

def process_frame(frame_data):
    # Fast decode with reduced checks
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return b''
    
    # Fast grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Optimized encode with faster parameters
    _, encoded = cv2.imencode('.jpg', gray, [
        cv2.IMWRITE_JPEG_QUALITY, 90,  # Reduced for speed
        cv2.IMWRITE_JPEG_OPTIMIZE, 1
    ])
    return encoded.tobytes()

def handle_client(conn):
    try:
        with conn:
            prev_time = time.time()
            frame_count = 0
            
            while True:
                # Receive header (4 bytes)
                size_header = conn.recv(4)
                if len(size_header) != 4: break
                frame_size = int.from_bytes(size_header, 'big')
                
                # Memoryview for zero-copy
                frame_data = bytearray(frame_size)
                view = memoryview(frame_data)
                total = 0
                while total < frame_size:
                    recv_size = conn.recv_into(view[total:], frame_size - total)
                    if recv_size == 0: break
                    total += recv_size
                
                # Process and send in parallel
                processed = process_frame(frame_data)
                conn.sendall(len(processed).to_bytes(4, 'big') + processed)
                
                # Monitoring
                frame_count += 1
                if frame_count % 30 == 0:
                    curr_time = time.time()
                    print(f"Server FPS: {30/(curr_time-prev_time):.2f}")
                    prev_time = curr_time

    except Exception as e:
        print(f"Client error: {e}")

if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)
        print(f"High-speed server listening on {PORT}")
        
        while True:
            conn, addr = s.accept()
            print(f"New connection: {addr[0]}")
            Thread(target=handle_client, args=(conn,), daemon=True).start()