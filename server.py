import socket
import cv2
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
import time

# Use TurboJPEG for faster encoding/decoding
from turbojpeg import TurboJPEG
jpeg = TurboJPEG()

parser = argparse.ArgumentParser(description='Optimized 4K Video Server')
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()

HOST = '0.0.0.0'
PORT = args.port
BUFFER_SIZE = 4 * 1024 * 1024
MAX_WORKERS = 4
FRAME_SIZE = (3840, 2160)

def process_frame(frame_data):
    try:
        # TurboJPEG decoding (5-10x faster than OpenCV)
        frame = jpeg.decode(frame_data)
        
        # Accelerated grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # TurboJPEG encoding
        return jpeg.encode(gray, quality=95)
    
    except Exception as e:
        print(f"Processing error: {e}")
        return b''

def handle_client(conn):
    with conn, ThreadPoolExecutor(max_workers=2) as executor:
        future = None
        while True:
            try:
                # Async receive
                size_header = conn.recv(4)
                if not size_header: break
                frame_size = int.from_bytes(size_header, 'big')
                
                # Zero-copy buffer
                frame_data = bytearray(frame_size)
                view = memoryview(frame_data)
                total = 0
                while total < frame_size:
                    recv_size = conn.recv_into(view[total:], frame_size - total)
                    if recv_size == 0: break
                    total += recv_size
                
                # Pipeline processing
                if future is not None:
                    processed_data = future.result()
                    conn.sendall(len(processed_data).to_bytes(4, 'big') + processed_data)
                
                # Submit next frame async
                future = executor.submit(process_frame, frame_data)
            
            except Exception as e:
                print(f"Connection error: {e}")
                break

if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Optimized server listening on {HOST}:{PORT}")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                conn, addr = s.accept()
                print(f"New connection: {addr}")
                executor.submit(handle_client, conn)