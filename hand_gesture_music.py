import cv2
import numpy as np
import sys
import math

# Import MediaPipe
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe loaded successfully!")
except (ImportError, AttributeError) as e:
    print(f"✗ MediaPipe not available: {e}")
    MEDIAPIPE_AVAILABLE = False
    mp_hands = None
    mp_drawing = None

# Import Pygame
import pygame

# Initialize Pygame
pygame.init()
pygame.mixer.init()
pygame.mixer.set_num_channels(1)

print("✓ Pygame initialized successfully!")

# Configuration
music_sound = None

def load_music_file(filepath):
    """Load a music file using pygame.mixer.music (better for longer files)"""
    try:
        pygame.mixer.music.load(filepath)
        
        # Get duration (this is approximate for MP3s)
        temp_sound = pygame.mixer.Sound(filepath)
        duration = temp_sound.get_length()
        
        print(f"✓ Successfully loaded: {filepath}")
        print(f"  Duration: {duration:.2f} seconds")
        
        return "USE_MUSIC_MODULE"  # Flag to use pygame.mixer.music
    except Exception as e:
        print(f"✗ Error loading music file: {e}")
        print("  Trying to create beep sound as fallback...")
        return None

def create_beep_sound():
    """Create a simple beep sound"""
    sample_rate = 22050
    duration = 0.5
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = np.sin(2 * np.pi * frequency * t)
    
    fade_samples = int(0.05 * sample_rate)
    wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
    wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    wave = (wave * 32767).astype(np.int16)
    stereo_wave = np.column_stack((wave, wave))
    
    return pygame.sndarray.make_sound(stereo_wave)

def play_music_continuous(sound, is_first_play=False):
    """Start or resume playing music"""
    if sound == "USE_MUSIC_MODULE":
        # Use pygame.mixer.music for music files
        if is_first_play or not pygame.mixer.music.get_busy():
            # Start from beginning if first play, otherwise unpause
            pygame.mixer.music.unpause()
    elif sound:
        # Use pygame.mixer.Sound for short sounds like beeps
        if not pygame.mixer.get_busy():
            sound.play(loops=-1)

def pause_music():
    """Pause the music (keeps position)"""
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.pause()

def stop_music():
    """Stop playing music completely (resets position)"""
    pygame.mixer.music.stop()
    pygame.mixer.stop()

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def detect_circle_hand_shape(hand_landmarks, frame_width, frame_height):
    """Detect if hand forms a circle shape (thumb and index finger touching)"""
    # Get landmark positions
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    wrist = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]
    
    # Convert to pixel coordinates
    thumb_tip_px = (int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height))
    index_tip_px = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))
    wrist_px = (int(wrist.x * frame_width), int(wrist.y * frame_height))
    index_mcp_px = (int(index_mcp.x * frame_width), int(index_mcp.y * frame_height))
    
    # Calculate distances
    thumb_index_distance = calculate_distance(thumb_tip_px, index_tip_px)
    hand_size = calculate_distance(wrist_px, index_mcp_px)
    
    # Threshold for "touching"
    touch_threshold = hand_size * 0.15
    
    # Check if thumb and index are touching
    fingers_touching = thumb_index_distance < touch_threshold
    
    return fingers_touching, thumb_tip_px, index_tip_px, thumb_index_distance

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Error: Could not open camera")
        print("  Make sure no other application is using the camera")
        return
    
    print("✓ Camera opened successfully!")
    
    # Initialize MediaPipe Hands if available
    hands = None
    if MEDIAPIPE_AVAILABLE:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ Hand tracking initialized!")
    
    # Tracking variables
    music_playing = False
    music_started = False  # Track if music has been started at all
    circle_detected_frames = 0
    circle_lost_frames = 0
    REQUIRED_FRAMES = 5  # Frames to start playing
    RELEASE_FRAMES = 5   # Frames to stop playing (adds stability)
    
    tracking_method = "MediaPipe" if MEDIAPIPE_AVAILABLE else "Not Available"
    
    print("\n" + "="*60)
    print("HAND GESTURE MUSIC TRACKER")
    print("="*60)
    print(f"Tracking Method: {tracking_method}")
    print(f"Music: {'Loaded' if music_sound and music_sound != create_beep_sound() else 'Default Beep'}")
    print("\nInstructions:")
    print("  • Make a circle with your hand (touch thumb and index finger)")
    print("  • Music plays while you hold the circle")
    print("  • Release to pause (keeps position)")
    print("  • Make circle again to resume")
    print("  • Press 'q' to quit")
    print("="*60 + "\n")
    
    if not MEDIAPIPE_AVAILABLE:
        print("⚠ WARNING: MediaPipe not available. Hand tracking will not work.")
        print("Please follow the setup instructions to install MediaPipe properly.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✗ Error: Could not read frame from camera")
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        circle_shape_detected = False
        
        if MEDIAPIPE_AVAILABLE and hands:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
                    )
                    
                    # Detect circle hand shape
                    is_circle, thumb_pos, index_pos, distance = detect_circle_hand_shape(
                        hand_landmarks, w, h
                    )
                    
                    if is_circle:
                        circle_shape_detected = True
                        # Draw circle indicator
                        cv2.circle(frame, thumb_pos, 15, (0, 255, 0), 3)
                        cv2.circle(frame, index_pos, 15, (0, 255, 0), 3)
                        cv2.line(frame, thumb_pos, index_pos, (0, 255, 0), 3)
                        
                        # Draw text
                        center_x = (thumb_pos[0] + index_pos[0]) // 2
                        center_y = (thumb_pos[1] + index_pos[1]) // 2
                        cv2.putText(frame, "CIRCLE!", (center_x - 50, center_y - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Update circle detection counter
        if circle_shape_detected:
            circle_detected_frames += 1
            circle_lost_frames = 0
        else:
            circle_lost_frames += 1
            if circle_lost_frames > RELEASE_FRAMES:
                circle_detected_frames = 0
        
        # Start music if circle held for enough frames
        if circle_detected_frames >= REQUIRED_FRAMES and not music_playing:
            music_playing = True
            if not music_started:
                # First time starting the music
                if music_sound == "USE_MUSIC_MODULE":
                    pygame.mixer.music.play(loops=-1)
                else:
                    play_music_continuous(music_sound)
                music_started = True
                print("▶️  Music started!")
            else:
                # Resume from pause
                play_music_continuous(music_sound)
                print("▶️  Music resumed!")
        
        # Pause music if circle lost for enough frames
        if circle_lost_frames > RELEASE_FRAMES and music_playing:
            music_playing = False
            pause_music()
            print("⏸️  Music paused!")
        
        # Display status
        if music_playing:
            status_text = "Playing music!"
            color = (0, 255, 0)
        elif circle_detected_frames > 0:
            status_text = f"Hold circle... {circle_detected_frames}/{REQUIRED_FRAMES}"
            color = (255, 255, 0)
        else:
            status_text = "Make a circle to play music"
            color = (255, 255, 255)
        
        # Draw status background
        cv2.rectangle(frame, (5, 5), (w-5, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (w-5, 100), (100, 100, 100), 2)
        
        # Draw status text
        cv2.putText(frame, status_text, (15, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.putText(frame, "Touch thumb + index | Release to pause", (15, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw tracking method at bottom
        cv2.putText(frame, f"Tracking: {tracking_method}", (15, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.imshow('Hand Gesture Music Tracker', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nExiting...")
            break
    
    # Cleanup
    stop_music()  # Make sure music stops when program exits
    cap.release()
    cv2.destroyAllWindows()
    if hands:
        hands.close()
    print("✓ Cleanup complete. Goodbye!")

if __name__ == "__main__":
    # Check if user provided a music file
    if len(sys.argv) > 1:
        music_file = sys.argv[1]
        music_sound = load_music_file(music_file)
        if not music_sound:
            print("Falling back to beep sound")
            music_sound = create_beep_sound()
    else:
        print("No music file provided.")
        print("Usage: python hand_gesture_music.py <path_to_music_file>")
        print("Using default beep sound.\n")
        music_sound = create_beep_sound()
    
    main()