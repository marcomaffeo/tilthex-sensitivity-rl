import cv2
from pathlib import Path
import sys

def extract_frame(video_path, timestamp_sec, output_path):
    path = Path(video_path)
    if not path.exists():
        print(f"Errore: Il file video '{video_path}' non esiste nel percorso corrente ({Path.cwd()}).")
        print("Assicurati di trovarti nella cartella giusta o specifica il percorso completo.")
        sys.exit(1)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Errore: OpenCV non riesce a decodificare il flusso video da '{video_path}'. Potrebbe essere un problema di codec.")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback di sicurezza se FPS non leggibili
        
    frame_idx = int(timestamp_sec * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Frame estratto con successo a {timestamp_sec}s (frame #{frame_idx}) e salvato in: {output_path}")
    else:
        print(f"Errore: Impossibile leggere il frame al secondo {timestamp_sec} (indice {frame_idx}).")
        
    cap.release()

if __name__ == "__main__":
    # Inserisci qui il nome corretto del file video (o il percorso relativo/assoluto)
    video_file = 'SPPO_p2p_perturbed.MP4'  # Oppure controlla se è in un'altra cartella, es. '../SPPO_p2p_perturbed.MP4'
    target_time = 24.0                     # Secondo esatto da estrarre
    output_image = 'extracted_frame.png'
    
    extract_frame(video_file, target_time, output_image)