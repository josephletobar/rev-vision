import argparse
import os
import cv2
import csv
import numpy as nps
import socket
import json
from vision.process_frame import ProcessFrame

def send_point(conn, x, y):
    msg = [float(x), float(y)]
    try:
        conn.sendall((json.dumps(msg) + "\n").encode())
    except (BrokenPipeError, ConnectionResetError):
        return False
    return True

def save_points_csv(write_dir, ball_cx, ball_cy, t_sec):
    write_path = f"{write_dir}/points.csv"
    file_exists = os.path.exists(write_path)

    with open(write_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["x", "y", "time_stamp"])

        writer.writerow([ball_cx, ball_cy, t_sec])
        f.flush()
        os.fsync(f.fileno())
    
def main():
    
    # set socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 5000))
    sock.listen(1)
    conn, _ = sock.accept()

    # set CSV at the start of each run
    TRACKING_OUTPUT = "outputs/points.csv"
    with open(TRACKING_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "time_stamp"])  # header

    # parse arguments
    parser = argparse.ArgumentParser(description="Lane Assist Video Processing")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Path to video file")
    group.add_argument("--websocket", type=int, help="WebSocket port for live mode")
    parser.add_argument("--output", type=str, help="Path to save outputs (optional)")
    args = parser.parse_args()


    # video processing
    if args.input: cap = cv2.VideoCapture(args.input)

    # set arguments
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(f"{args.output}/replay.mp4", fourcc, fps, (width, height))

    process_frame = ProcessFrame(out)

    try:

        if args.input:
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret: break
                if frame is None: continue

                result = process_frame.process_frame(frame)

                if result is None: continue
                x_smooth, y_smooth, t_sec = result

                save_points_csv("outputs", int(x_smooth), int(y_smooth), t_sec) 
                save_points_csv(args.output, int(x_smooth), int(y_smooth), t_sec)

                send_point(conn, x_smooth, y_smooth)

        elif args.websocket:
            recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            recv_sock.bind(("127.0.0.1", args.websocket))
            recv_sock.listen(1)
            frame_conn, _ = recv_sock.accept()

            while True:
                data = frame_conn.recv(65536)
                if not data:
                    break

                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                x_smooth, y_smooth = process_frame.process_frame(frame)

                save_points_csv("outputs", int(x_smooth), int(y_smooth), 0)
                save_points_csv(args.output, int(x_smooth), int(y_smooth), 0)

                send_point(conn, x_smooth, y_smooth)


    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Cleaning up gracefully...")

    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        try:
            conn.close()
            sock.close()
        except Exception:
            pass

        print("Done")

# python3 main.py --input test_videos/bowling.mp4
if __name__ == "__main__":
    main()
