import cv2
import numpy as np
import sys
import math

clicked_points = []
clone_image = None
is_red_mode = False  

def select_points(event, x, y, flags, param):
    global clicked_points, clone_image, is_red_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append({'coords': (x, y), 'is_red': is_red_mode})
        
        color = (0, 0, 255) if is_red_mode else (255, 0, 0)
        cv2.circle(clone_image, (x, y), 5, color, -1)
        cv2.imshow("Select Points", clone_image)
        
        if is_red_mode:
            is_red_mode = False
            print("Punto ROSSO registrato. Modalità colore tornata normale.")

def generate_chronophotography(video_path, output_path, timestamps, mode='dark', ghost_settings=None, bg_time=24.0):
    global clicked_points, clone_image, is_red_mode
    
    if ghost_settings is None:
        ghost_settings = {}
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(bg_time * fps))
    ret, bg_frame = cap.read()
    if not ret:
        print(f"Error: Could not read background frame at {bg_time}s.")
        sys.exit()
        
    base_frame = bg_frame.astype(np.float32)
    final_image = base_frame.copy()
    
    for t in sorted(timestamps):
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame at {t}s")
            continue
            
        frame_float = frame.astype(np.float32)
        opacity = ghost_settings.get(t, 1.0)
        
        if opacity < 1.0:
            frame_float = (frame_float * opacity) + (base_frame * (1.0 - opacity))
            
        if mode == 'dark':
            final_image = np.minimum(final_image, frame_float)
        elif mode == 'light':
            final_image = np.maximum(final_image, frame_float)

    cap.release()
    
    if final_image is not None:
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)
        
        clicked_points = []
        is_red_mode = False
        clone_image = final_image.copy()
        
        cv2.namedWindow("Select Points")
        cv2.setMouseCallback("Select Points", select_points)
        
        print("\n--- SELEZIONE PUNTI ---")
        print("Premi 'r' PRIMA di cliccare per far diventare ROSSO il prossimo punto.")
        print("Premi 'c' per resettare tutti i punti (Clear).")
        print("Premi 'Enter' quando hai finito.")
        print("Premi 'q' per uscire senza salvare.\n")
        
        cv2.imshow("Select Points", clone_image)
        
        while True:
            # waitKey aumentato a 20ms e imshow rimosso dal loop continuo
            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('c'):  
                clone_image = final_image.copy()
                clicked_points = []
                is_red_mode = False
                print("Punti resettati.")
                cv2.imshow("Select Points", clone_image)
            elif key == ord('r'):
                is_red_mode = True
                print("Modalità ROSSO attivata per il prossimo clic!")
            elif key == 13:
                break
            elif key == ord('q'):
                sys.exit()
                
        cv2.destroyAllWindows()
        
        grouped_points = []
        threshold = 30
        
        # Raggruppamento dei punti
        for i, pt_dict in enumerate(clicked_points):
            label = chr(65 + i)
            found = False
            for group in grouped_points:
                dist = math.hypot(pt_dict['coords'][0] - group['coords'][0], pt_dict['coords'][1] - group['coords'][1])
                if dist < threshold:
                    # Salviamo ogni etichetta con il suo stato colore
                    group['items'].append({'label': label, 'is_red': pt_dict['is_red']})
                    if pt_dict['is_red']:
                        group['has_red'] = True
                    found = True
                    break
            if not found:
                grouped_points.append({
                    'coords': pt_dict['coords'], 
                    'items': [{'label': label, 'is_red': pt_dict['is_red']}],
                    'has_red': pt_dict['is_red']
                })
                
        # Disegno su final_image
        for group in grouped_points:
            x, y = group['coords']
            has_red = group['has_red']
            
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.3
            thickness = 2
            start_x, start_y = (x + 30, y - 30)
            
            # 1. DISEGNO DEL PALLINO CENTRALE
            if has_red:
                point_color = (60, 60, 255)       
                # Alone bianco impostato a 5 (uguale al punto, quindi invisibile. Aumentalo per vederlo)
                cv2.circle(final_image, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)  
                cv2.circle(final_image, (x, y), 5, point_color, -1, cv2.LINE_AA)  
            else:
                cv2.circle(final_image, (x, y), 6, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(final_image, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            
            # 2. DISEGNO DEI CONTORNI DELLE LETTERE (per evitare accavallamenti)
            curr_x = start_x
            for idx, item in enumerate(group['items']):
                char = item['label']
                is_red = item['is_red']
                
                out_color = (255, 255, 255) if is_red else (0, 0, 0)
                out_thick = thickness + 5 if is_red else thickness + 3
                
                cv2.putText(final_image, char, (curr_x, start_y), font, font_scale, out_color, out_thick, cv2.LINE_AA)
                (w, h), _ = cv2.getTextSize(char, font, font_scale, thickness)
                curr_x += w
                
                if idx < len(group['items']) - 1:
                    eq_char = "="
                    cv2.putText(final_image, eq_char, (curr_x, start_y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
                    (w, h), _ = cv2.getTextSize(eq_char, font, font_scale, thickness)
                    curr_x += w
            
            # 3. DISEGNO DEL CORPO INTERNO DELLE LETTERE
            curr_x = start_x
            for idx, item in enumerate(group['items']):
                char = item['label']
                is_red = item['is_red']
                
                text_color = (60, 60, 255) if is_red else (255, 255, 255)
                cv2.putText(final_image, char, (curr_x, start_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
                (w, h), _ = cv2.getTextSize(char, font, font_scale, thickness)
                curr_x += w
                
                if idx < len(group['items']) - 1:
                    eq_char = "="
                    cv2.putText(final_image, eq_char, (curr_x, start_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    (w, h), _ = cv2.getTextSize(eq_char, font, font_scale, thickness)
                    curr_x += w

        cv2.imwrite(output_path, final_image)
        print(f"\nImmagine salvata con successo: {output_path}")
    else:
        print("Error: Could not extract any frames.")


# if __name__ == "__main__":
#     selected_times = [
#         12.0, 
#         17.0, 
#         19.6, 20.4, 24.0, 
#         38.5, 42.0
#     ]
    
#     ghost_opacities = {
#         12.0: 1.0, 
#         17.0: 0.5, 
#         19.6: 1.0,
#         20.4: 0.5, 
#         24.0: 1.0, 
#         38.5: 0.5, 
#         42.0: 1.0
#     }
#     generate_chronophotography(
#         'SPPO_att.MP4', 
#         'output_image.jpg', 
#         timestamps=selected_times, 
#         mode='dark',
#         ghost_settings=ghost_opacities,
#         bg_time=24.0 
#     )

if __name__ == "__main__":
        selected_times = [
            14.0, 17.5, 19.8,
            20.6, 24.0, 26.25,
            28.0, 33.5, 40.0    
        ]
        
        ghost_opacities = {
            14.0: 1.0, 
            17.5: 0.5, 
            19.8: 1.0,
            20.6: 0.5, 
            24.0: 1.0, 
            26.25: 0.5,
            28.0: 1.0, 
            33.5: 0.5, 
            40.0: 1.0,
        }
        
generate_chronophotography(
        'SPPO_p2p.MP4',
        'output_image.jpg', 
        timestamps=selected_times, 
        mode='dark',
        ghost_settings=ghost_opacities,
        bg_time=24.0 
    )