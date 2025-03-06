import socket
import cv2
import numpy as np
import argparse
from threading import Thread
import time

# Check CUDA availability
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"CUDA Available: {cuda_available}")

parser = argparse.ArgumentParser(description='Ultra Fast Video Server')
parser.add_argument('--port', type=int, required=True)
args = parser.parse_args()

UDP_IP = '0.0.0.0'
UDP_PORT = args.port
BUFFER_SIZE = 4 * 1024 * 1024
PROCESS_SIZE = (1920, 1080)  # Process at 1080p for speed

# Initialize GPU resources if available
if cuda_available:
    stream = cv2.cuda_Stream()

def process_frame(frame_data):
    try:
        # Fast decode with resize
        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
            
        # Resize for faster processing
        frame = cv2.resize(frame, PROCESS_SIZE)
        
        if cuda_available:
            # Upload to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame, stream=stream)
            
            # GPU accelerated processing
            gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY, stream=stream)
            
            # Download result
            result = gpu_gray.download(stream=stream)
            stream.waitForCompletion()
        else:
            # CPU fallback
            result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Fast encode with optimized parameters
        _, encoded = cv2.imencode('.jpg', result, [
            cv2.IMWRITE_JPEG_QUALITY, 85,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 1
        ])
        return encoded.tobytes()
    
    except Exception as e:
        print(f"Processing error: {e}")
        return None

def udp_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024 * 1024)
    
    seq_num = 0
    last_seq = -1
    fps_counter = 0
    last_time = time.time()
    
    print(f"UDP Server listening on {UDP_PORT}")
    
    while True:
        try:
            # Receive with buffer
            data, addr = sock.recvfrom(BUFFER_SIZE)
            current_seq = int.from_bytes(data[:4], 'big')
            
            # Skip old frames
            if current_seq <= last_seq:
                continue
                
            last_seq = current_seq
            
            # Process frame
            processed = process_frame(data[4:])
            
            if processed is not None:
                # Add sequence header and send back
                sock.sendto(current_seq.to_bytes(4, 'big') + processed, addr)
            
            # FPS counter
            fps_counter += 1
            if time.time() - last_time >= 1.0:
                print(f"Processing FPS: {fps_counter}")
                fps_counter = 0
                last_time = time.time()

        except Exception as e:
            print(f"UDP error: {e}")

if __name__ == '__main__':
    Thread(target=udp_server, daemon=True).start()
    
    # Keep main thread alive
    while True:
        time.sleep(1)