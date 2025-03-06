import socket
import cv2
import numpy as np
import argparse
from threading import Thread, Lock
import time
import queue

parser = argparse.ArgumentParser(description='Optimized Video Server')
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()

HOST = '0.0.0.0'
PORT = args.port
BUFFER_SIZE = 4 * 1024 * 1024
MAX_QUEUE_SIZE = 2  # Prevent memory bloat

# Shared processing queue with thread-safe access
processing_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
queue_lock = Lock()

def process_worker():
    while True:
        try:
            conn, frame_data = processing_queue.get()
            if frame_data is None:
                break
            
            # Decode with optimized parameters
            frame = cv2.imdecode(
                np.frombuffer(frame_data, np.uint8), 
                cv2.IMREAD_GRAYSCALE  # Direct grayscale decode
            )
            
            if frame is not None:
                # Encode with fast settings
                _, encoded = cv2.imencode('.jpg', frame, [
                    cv2.IMWRITE_JPEG_QUALITY, 85,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 1
                ])
                response = encoded.tobytes()
            else:
                response = b''
            
            with queue_lock:
                conn.sendall(len(response).to_bytes(4, 'big') + response)
        
        except Exception as e:
            print(f"Processing error: {e}")

# Start worker threads
for _ in range(4):  # Match CPU core count
    Thread(target=process_worker, daemon=True).start()

def handle_client(conn):
    try:
        with conn:
            prev_time = time.time()
            frame_count = 0
            
            while True:
                # Get frame size
                size_header = conn.recv(4)
                if len(size_header) != 4: 
                    break
                frame_size = int.from_bytes(size_header, 'big')
                
                # Buffer pre-allocation
                frame_data = bytearray(frame_size)
                view = memoryview(frame_data)
                total = 0
                while total < frame_size:
                    recv_size = conn.recv_into(view[total:], min(4096, frame_size - total))
                    if recv_size == 0: 
                        break
                    total += recv_size
                
                # Async processing
                try:
                    processing_queue.put_nowait((conn, bytes(frame_data)))
                except queue.Full:
                    print("Queue full - dropping frame")
                
                # FPS reporting
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
        print(f"Optimized server listening on {PORT}")
        
        while True:
            conn, addr = s.accept()
            print(f"New connection: {addr[0]}")
            Thread(target=handle_client, args=(conn,), daemon=True).start()