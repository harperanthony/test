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

# Pre-allocate reusable buffers
HEADER_SIZE = 4
encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1]

def process_frame(frame_data):
    # Fast decode with fixed buffer
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return None
    
    # Optimized grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Faster encode with pre-defined params
    success, encoded = cv2.imencode('.jpg', gray, encode_params)
    return encoded.tobytes() if success else None

def handle_client(conn):
    try:
        prev_time = time.time()
        frame_count = 0
        recv_buffer = bytearray()
        send_buffer = bytearray()
        
        while True:
            # Receive header
            while len(recv_buffer) < HEADER_SIZE:
                chunk = conn.recv(BUFFER_SIZE)
                if not chunk: break
                recv_buffer += chunk
            
            if len(recv_buffer) < HEADER_SIZE: break
            frame_size = int.from_bytes(recv_buffer[:HEADER_SIZE], 'big')
            recv_buffer = recv_buffer[HEADER_SIZE:]
            
            # Receive frame data
            while len(recv_buffer) < frame_size:
                chunk = conn.recv(BUFFER_SIZE)
                if not chunk: break
                recv_buffer += chunk
            
            if len(recv_buffer) < frame_size: break
            
            # Process and send
            processed = process_frame(recv_buffer[:frame_size])
            recv_buffer = recv_buffer[frame_size:]
            
            if processed:
                # Prepare send buffer
                send_buffer = len(processed).to_bytes(HEADER_SIZE, 'big') + processed
                conn.sendall(send_buffer)
            
            # FPS counter
            frame_count += 1
            if frame_count % 30 == 0:
                curr_time = time.time()
                print(f"Server FPS: {30/(curr_time-prev_time):.2f}")
                prev_time = curr_time

    except Exception as e:
        print(f"Client error: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)
        print(f"Server listening on {PORT}")
        
        while True:
            conn, addr = s.accept()
            print(f"New connection: {addr[0]}")
            Thread(target=handle_client, args=(conn,), daemon=True).start()