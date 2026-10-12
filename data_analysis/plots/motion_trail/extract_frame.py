import cv2
import sys
from pathlib import Path

# Variabili globali per la selezione del ROI (Region of Interest)
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
crop_box = None
clone = None

def mouse_crop(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, crop_box, clone
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        crop_box = (min(ix, fx), min(iy, fy), abs(fx - ix), abs(fy - iy))

def extract_and_crop_frame(video_path, timestamp_sec, output_path):
    global crop_box, clone, drawing
    
    path = Path(video_path)
    if not path.exists():
        print(f"Errore: Il file video '{video_path}' non esiste nel percorso corrente.")
        sys.exit(1)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Errore: OpenCV non riesce a decodificare il flusso video da '{video_path}'.")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        
    frame_idx = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Errore: Impossibile leggere il frame al secondo {timestamp_sec} (indice {frame_idx}).")
        sys.exit(1)
        
    clone = frame.copy()
    cv2.namedWindow("Trascina per ritagliare - Premi Enter per confermare", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Trascina per ritagliare - Premi Enter per confermare", mouse_crop)
    
    print("\n--- ISTRUZIONI DI RITAGLIO ---")
    print("1. Clicca e trascina il mouse sull'immagine per selezionare l'area da ritagliare.")
    print("2. Premi 'r' per azzerare la selezione e rifarla.")
    print("3. Premi 'Enter' per confermare e salvare.")
    print("4. Premi 'q' per uscire senza salvare.\n")
    
    while True:
        display_image = clone.copy()
        
        # Disegna il rettangolo in tempo reale mentre trascini il mouse
        if drawing or crop_box is not None:
            if drawing:
                rx, ry, rw, rh = min(ix, fx), min(iy, fy), abs(fx - ix), abs(fy - iy)
            else:
                rx, ry, rw, rh = crop_box
                
            if rw > 0 and rh > 0:
                cv2.rectangle(display_image, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
                
        cv2.imshow("Trascina per ritagliare - Premi Enter per confermare", display_image)
        key = cv2.waitKey(20) & 0xFF
        
        if key == ord('r'):
            crop_box = None
            drawing = False
            print("Selezione resettata.")
        elif key == 13: # Tasto Enter
            if crop_box is not None and crop_box[2] > 0 and crop_box[3] > 0:
                break
            else:
                print("Attenzione: Seleziona prima un'area valida trascinando il mouse!")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("Operazione annullata.")
            sys.exit(0)
            
    cv2.destroyAllWindows()
    
    # Esegue il ritaglio e salva in formato PNG lossless (qualità massima)
    rx, ry, rw, rh = crop_box
    cropped_frame = frame[ry:ry+rh, rx:rx+rw]
    
    cv2.imwrite(output_path, cropped_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Frame ritagliato con successo e salvato in: {output_path}")

if __name__ == "__main__":
    # Configura qui il file video, il secondo esatto e l'immagine di output
    video_file = 'SPPO_p2p_perturbed.MP4'  # Modifica se necessario
    target_time = 24.0                      # Secondo del frame da estrarre
    output_image = 'cropped_frame.png'      # Nome del file finale
    
    extract_and_crop_frame(video_file, target_time, output_image)