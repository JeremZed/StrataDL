from stratadl.core.utils import file as StrataDL_file
from stratadl.core.preprocessing import image as StrataDL_image
import cv2
import os

import logging

EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg'}

def split_to_frames(input_folder:str, output_folder:str, frame_skip=5, resize_ratio=None):
    """
    Permet de découper une video en images
    """
    # Création du dossier de destination
    StrataDL_file.create_directory(output_folder)

    # On récupère les vidéos
    video_paths = StrataDL_file.get_files(input_folder, EXTENSIONS, recursive=True)

    if not video_paths:
        raise Exception(f"Aucune vidéo trouvée dans {input_folder}")

    logging.debug(f"{len(video_paths)} vidéo(s) trouvée(s)")

    saved_hashes = set()
    for idx, video_path in enumerate(video_paths, 1):
        logging.debug(f"\n[{idx}/{len(video_paths)}] Traitement : {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"  ⚠ Impossible d'ouvrir la vidéo : {video_path}")
            continue

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:

                if resize_ratio is not None:
                    frame = StrataDL_image.resize_image(frame, ratio=resize_ratio)

                # Calculer hash
                frame_hash = StrataDL_image.hash_frame(frame)
                if frame_hash not in saved_hashes:
                    saved_hashes.add(frame_hash)
                    filename = os.path.join(output_folder, f"{frame_hash}.jpg")
                    cv2.imwrite(filename, frame)
                    saved_count += 1

            frame_count += 1

        cap.release()
        logging.debug(f"  ✓ {saved_count} frames uniques sauvegardées ({frame_count} frames analysées)")

    logging.debug(f"\n{'='*60}")
    logging.debug(f"Extraction terminée : {len(saved_hashes)} frames uniques sauvegardées dans {output_folder}")
    logging.debug(f"{'='*60}")

