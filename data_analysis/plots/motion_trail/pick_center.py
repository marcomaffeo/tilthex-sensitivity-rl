#!/usr/bin/env python3
"""
pick_center.py

Apre un video e fa scegliere a mano, con un click, il punto che fara' da centro
del body frame nel video prodotto da overlay_body_axes.py (--center).

NOTA: serve un display vero (si apre una finestra), quindi va lanciato sul Mac
in un terminale normale, non da qui.

Mostra un fotogramma ogni --interval secondi, nell'intervallo --start/--end
(secondi nel VIDEO). Per ogni fotogramma:
  click sinistro = segna il centro del drone li', passa al fotogramma successivo
  s               = salta questo fotogramma (nessun punto)
  q               = esci e salva quello che e' stato scelto finora

Salva i punti in un file di testo, "t:x:y,t:x:y,..." (t = secondi nel video),
che overlay_body_axes.py legge passandolo direttamente a --center (gli basta
il percorso del file).

Uso:
    python3 pick_center.py SPPO_att.MP4 center_points.txt --start 22 --end 36 --interval 1
"""
import argparse
import sys

import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("out", help="file di testo dove salvare i punti scelti")
    ap.add_argument("--start", type=float, default=0.0, help="inizio (secondi nel video)")
    ap.add_argument("--end", type=float, default=None, help="fine (secondi nel video)")
    ap.add_argument("--interval", type=float, default=2.0,
                     help="passo fra un fotogramma mostrato e il successivo (s)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Non riesco ad aprire {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    end = args.end if args.end is not None else duration

    points = []
    clicked = {}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["pt"] = (x, y)

    win = "pick_center - click sul centro del drone, s=salta, q=esci"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    t = args.start
    quit_all = False
    while t <= end and not quit_all:
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        clicked.clear()
        base = frame.copy()
        cv2.putText(base, f"t={t:.2f}s  click=scegli centro, s=salta, q=esci",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        disp = base
        while True:
            cv2.imshow(win, disp)
            if "pt" in clicked:
                x, y = clicked["pt"]
                disp = base.copy()
                cv2.circle(disp, (x, y), 6, (0, 0, 255), -1)
                cv2.imshow(win, disp)
                cv2.waitKey(250)
                points.append((t, x, y))
                print(f"  t={t:.2f}s -> ({x},{y})")
                break
            key = cv2.waitKey(30) & 0xFF
            if key == ord('s'):
                print(f"  t={t:.2f}s -> saltato")
                break
            if key == ord('q'):
                quit_all = True
                break
        t += args.interval

    cv2.destroyAllWindows()
    cap.release()

    if not points:
        print("Nessun punto scelto, file non scritto.")
        return

    spec = ",".join(f"{t:.2f}:{x}:{y}" for t, x, y in points)
    with open(args.out, "w") as f:
        f.write(spec + "\n")
    print(f"\nSalvati {len(points)} punti in {args.out}")
    print(f"Usa:  --center {args.out}")


if __name__ == "__main__":
    main()
