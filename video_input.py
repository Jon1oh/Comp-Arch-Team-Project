import cv2
import os
import argparse

def get_video_stream(video_path):
    """
    Safely opens the video file and yields frames one by one.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break 
            
            # Hand the raw pixel data out to whoever called this function
            yield frame 
            
    finally:
        # Guarantee the file is released from memory when the video ends or on error
        cap.release()

def logic_manager(frame_stream):
    """
    A generic logic manager. 
    Receives the raw video frames and applies whatever custom processing is needed.
    """
    print("Logic manager received the stream. Starting processing...")
    frame_count = 0
    
    # Iterate directly over the generator stream
    for frame in frame_stream:
        frame_count += 1
        
        # --- Your custom logic or downstream functions go here ---
        # The 'frame' variable contains the raw image matrix.
        
        if frame_count % 30 == 0:
            print(f"Logic Manager processed {frame_count} frames...")
            
    print(f"Logic Manager finished. Total frames processed: {frame_count}")

def main():
    parser = argparse.ArgumentParser(description="Ingest a video and pass to logic manager.")
    parser.add_argument("video_path", type=str, help="The path to the input video file")
    args = parser.parse_args()

    try:
        # 1. Create the stream generator
        stream = get_video_stream(args.video_path)
        
        # 2. Pass the stream directly into the logic manager
        logic_manager(stream)
        
    except (FileNotFoundError, IOError) as e:
        print(f"Pipeline Halted: {e}")

if __name__ == "__main__":
    main()