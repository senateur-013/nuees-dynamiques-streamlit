import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nuees_dynamiques import NueesDynamiques

st.set_page_config(page_title="Nuées Dynamiques", layout="centered")

st.title("Méthode des Nuées Dynamiques (E. Diday, 1971)")
st.write("Classification automatique par étalons dynamiques")

uploaded_file = st.file_uploader("Importer un fichier CSV (2 colonnes minimum)", type=["csv"])

K = st.slider("Nombre de classes (K)", 2, 10, 3)
ni = st.slider("Nombre d'étalons par classe (ni)", 2, 20, 5)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        X = df.values

        if X.shape[1] < 2:
            st.error("Le fichier doit contenir au moins deux colonnes numériques.")
        else:
            nd = NueesDynamiques(K=K, ni=ni)
            classes = nd.fit(X)

            st.success("Clustering terminé avec succès !")

            fig, ax = plt.subplots()

            couleurs = ["red", "blue", "green", "orange", "purple", "brown", "pink"]

            for i, Ci in enumerate(classes):
                Ci = np.array(Ci)
                if Ci.size > 0:
                    ax.scatter(Ci[:, 0], Ci[:, 1], label=f"Classe {i+1}",
                               color=couleurs[i % len(couleurs)])

                Ei = np.array(nd.E[i])
                ax.scatter(Ei[:, 0], Ei[:, 1], marker="x", s=120, color="black")

            ax.set_title("Résultat du Clustering")
            ax.legend()
            st.pyplot(fig)

            st.subheader("Taille des classes :")
            for i, Ci in enumerate(classes):
                st.write(f"Classe {i+1}: {len(Ci)} éléments")

    except Exception as e:
        st.error(f"Erreur lors de l'analyse du fichier : {e}")

