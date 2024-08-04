import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt


def estimer_flot_optique(chemin_video, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
                                   poly_sigma=1.2, flags=0, pas_images=15):
    """
    Estime le flot optique sur une vidéo en utilisant la méthode de Farneback et retourne un dictionnaire.

    :param chemin_video: Chemin vers la vidéo à analyser.
    :param pyr_scale: Facteur de réduction de l'image pyramidale.
    :param levels: Nombre de niveaux de la pyramide.
    :param winsize: Taille de la fenêtre de moyenne.
    :param iterations: Nombre d'itérations par niveau.
    :param poly_n: Taille du voisinage utilisé pour le polynôme.
    :param poly_sigma: Écart-type du polynôme.
    :param flags: Drapeaux d'option.
    :param pas_images: Nombre d'images à sauter entre chaque estimation.

    :return: Un dictionnaire contenant la magnitude et la direction des vecteurs de flot optique pour chaque image traitée.
    """

    capture = cv2.VideoCapture(chemin_video)

    if not capture.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo")
        return None

    ret, ancienne_image = capture.read()
    if not ret:
        print("Erreur: Impossible de lire la première image")
        return None

    ancienne_image_gris = cv2.cvtColor(ancienne_image, cv2.COLOR_RGB2GRAY)

    compteur_images = 0

    resultats = []

    while True:
        ret, image = capture.read()
        if not ret:
            break

        compteur_images += 1

        if compteur_images % pas_images != 0:
            continue

        image_gris = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(ancienne_image_gris, image_gris, None, pyr_scale, levels, winsize,
                                            iterations, poly_n, poly_sigma, flags)

        magnitude, direction = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        resultats.append(
            (np.max(magnitude),
             np.mean(magnitude),
             np.mean(direction)))

        plt.imsave('ResultatFarneback/mag' + str(compteur_images) + '.png', magnitude, cmap='gray')
        plt.imsave('ResultatFarneback/dir' + str(compteur_images) + '.png', direction, cmap='gray')

        # Mise à jour pour la prochaine itération
        ancienne_image_gris = image_gris.copy()

    capture.release()
    cv2.destroyAllWindows()

    return resultats

def enregistrer_resultats_csv(resultats, fichier_csv):
    with open(fichier_csv, mode='w', newline='') as fichier:
        writer = csv.writer(fichier)
        writer.writerow(['magnitude_max', 'magnitude_moyenne', 'direction_moyenne'])
        writer.writerows(resultats)

res = estimer_flot_optique('1H_side1.mp4')

if res is not None:
    enregistrer_resultats_csv(res, 'resultats_flot_optique2.csv')
    print("resultat enregistré")
