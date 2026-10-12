#!/usr/bin/env python3
"""
overlay_body_axes.py

Sovrappone a un video la terna degli assi del corpo del drone (x rosso, y verde,
z blu), ricostruita dal quaternione di assetto salvato in data.npz. Il centro
della terna va scelto a mano (a occhio, o con pick_center.py), nessun tracking
automatico.

Uso minimo (centro fisso, assetto stimato):
    python3 overlay_body_axes.py SPPO_att.MP4 rpy_nominal/SPPO_14_.../data.npz out.mp4 \
        --center 1550,920 --t0 3.5 --start 22 --end 36

--start/--end TAGLIANO l'output a quella finestra di secondi nel video (il file
prodotto dura solo quello).

--draw-start/--draw-end invece lasciano il video intero e disegnano gli assi solo
in quella finestra di secondi nel video (fuori da li' il fotogramma resta
originale, senza terna). E' la combinazione giusta per "voglio vedere l'intero
volo ma gli assi solo fra il secondo 22 e il 36":
    python3 overlay_body_axes.py SPPO_att.MP4 <npz> out.mp4 \
        --center center_points.txt --t0 14 --cam-rot="-90,0,0" \
        --draw-start 22 --draw-end 36

--center accetta tre forme:
  - "x,y"              punto fisso in pixel
  - "t:x:y,t:x:y,..."   keyframe interpolati linearmente, t = secondi nel VIDEO
  - il percorso di un file che contiene una delle due righe sopra (il formato
    prodotto da pick_center.py)

--t0 e' l'istante del video (in secondi) che corrisponde a t=0 nel file .npz: il
video ha margine prima/dopo il volo, va individuato a occhio (es. l'istante del
decollo).

--cam-rot "roll,pitch,yaw" (gradi) e' una rotazione fissa applicata alla terna
prima di proiettarla sul piano immagine, per compensare l'orientamento della
camera rispetto al world frame: si parte da "0,0,0" e si aggiusta finche' la
terna non ruota in modo plausibile rispetto a quel che si vede nel video. Non e'
una calibrazione vera, solo un allineamento a occhio.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np


def quat_wxyz_to_matrix(q):
    """Quaternione (w, x, y, z) -> matrice di rotazione 3x3 body-to-world."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def euler_deg_to_matrix(roll, pitch, yaw):
    r, p, y = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def parse_center(spec):
    """Ritorna una funzione t_video -> (x, y) in pixel, interpolando fra keyframe.

    spec puo' essere "x,y", "t:x:y,t:x:y,..." oppure il percorso di un file che
    contiene una di queste due righe (es. l'output di pick_center.py)."""
    if os.path.isfile(spec):
        with open(spec) as f:
            spec = f.read().strip()

    if ":" not in spec:
        x, y = (float(v) for v in spec.split(","))
        return lambda t: (x, y)

    kfs = []
    for chunk in spec.split(","):
        t, x, y = (float(v) for v in chunk.split(":"))
        kfs.append((t, x, y))
    kfs.sort(key=lambda k: k[0])
    ts = np.array([k[0] for k in kfs])
    xs = np.array([k[1] for k in kfs])
    ys = np.array([k[2] for k in kfs])

    def interp(t):
        return float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys))

    return interp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--center", default=None,
                     help="x,y fisso, keyframe 't:x:y,...' (t=secondi nel video), o file")
    ap.add_argument("--t0", type=float, default=0.0,
                     help="istante del video (s) corrispondente a t=0 nel .npz")
    ap.add_argument("--axis-len", type=float, default=120.0, help="lunghezza assi in pixel")
    ap.add_argument("--label-offset", type=float, default=30.0,
                     help="distanza extra (pixel) delle lettere x/y/z oltre la punta della freccia")
    ap.add_argument("--label-scale", type=float, default=0.9,
                     help="dimensione del carattere delle lettere x/y/z")
    ap.add_argument("--cam-rot", default="0,0,0",
                     help="roll,pitch,yaw (gradi) applicati prima della proiezione")
    ap.add_argument("--ref", action="store_true",
                     help="usa quat_ref (riferimento) invece di quat (assetto stimato)")
    ap.add_argument("--start", type=float, default=None,
                     help="output: inizio in secondi nel video (TAGLIA il video, non solo gli assi)")
    ap.add_argument("--end", type=float, default=None,
                     help="output: fine in secondi nel video (TAGLIA il video, non solo gli assi)")
    ap.add_argument("--draw-start", type=float, default=None,
                     help="disegna gli assi solo da questo secondo del video in poi (il video resta intero)")
    ap.add_argument("--draw-end", type=float, default=None,
                     help="disegna gli assi solo fino a questo secondo del video (il video resta intero)")
    ap.add_argument("--no-audio", action="store_true", help="non ricopiare l'audio originale")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    t_npz = np.asarray(data["t"], dtype=float)
    t_npz = t_npz - t_npz[0]
    quat = np.asarray(data["quat_ref" if args.ref else "quat"], dtype=float)
    duration_npz = t_npz[-1]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Non riesco ad aprire {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(round(args.start * fps)) if args.start else 0
    end_frame = int(round(args.end * fps)) if args.end else None
    if args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    center_fn = parse_center(args.center) if args.center else (lambda t: (width / 2, height / 2))
    cam_rot = euler_deg_to_matrix(*(float(v) for v in args.cam_rot.split(",")))

    silent_out = args.out + ".silent.mp4" if not args.no_audio else args.out
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_out, fourcc, fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"Non riesco ad aprire il writer per {silent_out}")

    axis_specs = [("x", (0, 0, 255)), ("y", (0, 255, 0)), ("z", (255, 0, 0))]

    frame_idx = start_frame
    n_written = 0
    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break
        t_video = frame_idx / fps
        t_flight = t_video - args.t0

        in_draw_window = True
        if args.draw_start is not None and t_video < args.draw_start:
            in_draw_window = False
        if args.draw_end is not None and t_video > args.draw_end:
            in_draw_window = False

        if in_draw_window and 0.0 <= t_flight <= duration_npz:
            idx = min(int(np.searchsorted(t_npz, t_flight)), len(t_npz) - 1)
            R_body = quat_wxyz_to_matrix(quat[idx])
            R_draw = cam_rot @ R_body
            cx, cy = center_fn(t_video)
            center_pt = (int(round(cx)), int(round(cy)))
            cv2.circle(frame, center_pt, 4, (255, 255, 255), -1)
            for col_idx, (label, color) in enumerate(axis_specs):
                axis_world = R_draw[:, col_idx]
                dx, dy = axis_world[0], axis_world[1]
                tip = (int(round(cx + args.axis_len * dx)),
                       int(round(cy - args.axis_len * dy)))
                cv2.arrowedLine(frame, center_pt, tip, color, 3, tipLength=0.2)
                # etichetta spostata lungo la stessa direzione della freccia, non con
                # un offset fisso in x/y (altrimenti si legge male se la freccia e'
                # verticale o punta a sinistra)
                label_dist = args.axis_len + args.label_offset
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, args.label_scale, 2)
                label_pt = (int(round(cx + label_dist * dx - text_w / 2)),
                            int(round(cy - label_dist * dy + text_h / 2)))
                cv2.putText(frame, label, label_pt,
                            cv2.FONT_HERSHEY_SIMPLEX, args.label_scale, color, 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1
        n_written += 1

    cap.release()
    writer.release()

    if args.no_audio:
        print(f"Fatto: {args.out} ({n_written} fotogrammi)")
        return

    cmd = ["ffmpeg", "-y", "-i", silent_out, "-i", args.video,
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
           "-shortest", args.out]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Mux audio fallito, resta il file senza audio:", silent_out)
        print(res.stderr[-800:])
    else:
        os.remove(silent_out)
        print(f"Fatto: {args.out} ({n_written} fotogrammi)")


if __name__ == "__main__":
    main()
