from queue import Queue
from threading import Event, Thread
from time import sleep, time
import cv2
import numpy as np

# =====================================
DELAY_S = 5
TELECAMERA = 1
AUTO_FOCUS = 1 # 1 abilitato    -    0 didabilitato
FOCUS = 45
# =====================================



cap = cv2.VideoCapture(TELECAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
cap.set(cv2.CAP_PROP_AUTOFOCUS, AUTO_FOCUS)
cap.set(cv2.CAP_PROP_FOCUS, FOCUS)
#//cap.set(cv2.CAP_PROP_SHARPNESS, 200)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

frames: Queue[tuple[np.ndarray, float]] = Queue()

stop_event = Event()
isFullscreen = False


def capture_frames() -> None:
    while not stop_event.is_set():
        ret, frame = cap.read()

        #? apply filters
        fgmask = fgbg.apply(frame)
        gaussian_blur = cv2.GaussianBlur(frame, (9, 9), 10.0)
        sharpened_frame = cv2.addWeighted(frame, 1.5, gaussian_blur, -0.5, 0)
        _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        frame = np.where(fgmask[:,:,None] == 255, sharpened_frame, frame)
        
        if not ret:
            stop_event.set()
        
        current_time = time()
        frames.put((frame, current_time))

t1 = Thread(target=capture_frames)
t1.start()

try:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #//print(cap.get(cv2.CAP_PROP_FPS))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #//print(frame_width, frame_height)

    black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    cv2.namedWindow("Feed Telecamera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Feed Telecamera", 600, 337)

    while not stop_event.is_set():
        if not frames.empty():
            
            frame, _ = frames.get()
            cv2.imshow('Feed Telecamera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('d') or key == ord('D'):
            cv2.imshow('Feed Telecamera', black_frame)
            break
        if key == ord("f") or key == ord('F'):
            isFullscreen = not isFullscreen
            cv2.setWindowProperty("Feed Telecamera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if isFullscreen else cv2.WINDOW_NORMAL)
        if key == ord('q') or key == ord('Q'):
            stop_event.set()

    while not stop_event.is_set():
        if not frames.empty():
            frame, frame_time = frames.get()
            remaining_delay = DELAY_S - (time() - frame_time)
            if remaining_delay > 0:
                sleep(remaining_delay)
            cv2.imshow('Feed Telecamera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("f") or key == ord('F'):
            isFullscreen = not isFullscreen
            cv2.setWindowProperty("Feed Telecamera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if isFullscreen else cv2.WINDOW_NORMAL)
        if key == ord('q') or key == ord('Q'):
            break
finally:
    stop_event.set()
    t1.join()
    cap.release()
    cv2.destroyAllWindows()
