import cv2
import numpy as np
import csv

def estimer_flot_optique(chemin_video, max_coins=100, niveau_qualite=0.3, distance_min=7, 
                         taille_bloc=7, taille_fenetre=(15, 15), niveau_max=2, 
                         criteres=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                         pas_images=15):
    """
    Estime le flot optique sur une vidéo en utilisant la méthode de Lucas-Kanade et retourne un dictionnaire.

    :param chemin_video: Chemin vers la vidéo à analyser.
    :param max_coins: Nombre maximum de points caractéristiques à détecter.
    :param niveau_qualite: Niveau de qualité pour la détection des points.
    :param distance_min: Distance minimale entre deux points caractéristiques.
    :param taille_bloc: Taille du bloc utilisé pour la détection des points.
    :param taille_fenetre: Taille de la fenêtre utilisée pour l'estimation du flot optique.
    :param niveau_max: Nombre maximum de niveaux pyramidaux pour l'estimation.
    :param criteres: Critères d'arrêt pour l'algorithme de flot optique.
    :param pas_images: Nombre d'images à sauter entre chaque estimation.
    
    :return: Un dictionnaire contenant les points initiaux, les vecteurs de déplacement, 
             la magnitude et la direction des vecteurs de flot optique.
    """

    capture = cv2.VideoCapture(chemin_video)

    if not capture.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo")
        return None

    params_points = dict(maxCorners=max_coins,
                         qualityLevel=niveau_qualite,
                         minDistance=distance_min,
                         blockSize=taille_bloc)

    params_lk = dict(winSize=taille_fenetre,
                     maxLevel=niveau_max,
                     criteria=criteres)

    ret, ancienne_image = capture.read()
    if not ret:
        print("Erreur: Impossible de lire la première image")
        return None

    ancienne_image_gris = cv2.cvtColor(ancienne_image, cv2.COLOR_RGB2GRAY)

    points_initiaux = cv2.goodFeaturesToTrack(ancienne_image_gris, mask=None, **params_points)

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

        points_suivis, statut, err = cv2.calcOpticalFlowPyrLK(ancienne_image_gris, image_gris, points_initiaux, None, **params_lk)

        points_suivis_valides = points_suivis[statut == 1]
        points_initiaux_valides = points_initiaux[statut == 1]

        u = points_suivis_valides[:, 0] - points_initiaux_valides[:, 0]
        v = points_suivis_valides[:, 1] - points_initiaux_valides[:, 1]
        magnitude = np.sqrt(u**2 + v**2)
        direction = np.arctan2(v, u)

        for i in range(len(points_initiaux_valides)):
            resultats.append((
                points_initiaux_valides[i, 0], points_initiaux_valides[i, 1], 
                u[i], v[i],                               
                magnitude[i],                             
                direction[i]                              
            ))

        # Mise à jour pour la prochaine itération
        ancienne_image_gris = image_gris.copy()
        points_initiaux = points_suivis_valides.reshape(-1, 1, 2)

    capture.release()
    cv2.destroyAllWindows()

    return resultats

def enregistrer_resultats_csv(resultats, fichier_csv):
    with open(fichier_csv, mode='w', newline='') as fichier:
        writer = csv.writer(fichier)
        writer.writerow(['x_initial', 'y_initial', 'u_deplacement', 'v_deplacement', 'magnitude', 'direction'])
        writer.writerows(resultats)

res = estimer_flot_optique('1H_side1.mp4', max_coins=150, pas_images=10, taille_fenetre=(50, 50))

if res is not None:
    enregistrer_resultats_csv(res, 'resultats_flot_optique.csv')
    print("resultat enregistré")
