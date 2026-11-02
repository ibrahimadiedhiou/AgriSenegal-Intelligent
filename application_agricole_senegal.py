# -*- coding: utf-8 -*-
"""
APPLICATION AGRICOLE INTELLIGENTE - SENEGAL
==========================================
Application Streamlit de diagnostic territorial, choix des cultures,
calcul économique et optimisation sous contraintes.

Lancement :
    pip install -r requirements.txt
    streamlit run application_agricole_senegal.py

DONNEES ET PRUDENCE :
- Les profils régionaux sont des orientations agroécologiques qualitatives.
- Les paramètres de prix, rendements, coûts et besoins en eau/travail sont
  des hypothèses modifiables, à remplacer par des références locales.
- Le module communal fonctionne avec un fichier CSV importé par l'utilisateur.
  Le fichier de communes fourni est un modèle vierge : aucune liste communale
  exhaustive ni donnée pédologique communale n'est inventée.
- L'optimisation calcule une solution pour les hypothèses saisies, pas une
  prescription agronomique certifiée.
"""

from __future__ import annotations
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="AgriSénégal Intelligent", page_icon="🌱",
                   layout="wide", initial_sidebar_state="expanded")

# -------------------------------------------------------------------
# DONNEES TERRITORIALES REGIONALES : profils qualitatifs de départ
# -------------------------------------------------------------------
REGIONS = {
    "Dakar": {
        "zone": "Niayes / périurbain",
        "forces": "Proximité des marchés, savoir-faire horticole, logistique et demande urbaine.",
        "faiblesses": "Pression foncière, salinité possible, coût et concurrence sur l'eau.",
        "sol": "Sols sableux à texture variable ; vérifier salinité, matière organique et pH.",
        "cultures": ["Oignon", "Tomate", "Piment", "Chou", "Laitue"],
        "conseils": "Maraîchage périurbain raisonné, goutte-à-goutte, paillage et contrats de vente."
    },
    "Diourbel": {
        "zone": "Bassin arachidier",
        "forces": "Systèmes agricoles intégrés, marchés régionaux et expérience des cultures pluviales.",
        "faiblesses": "Pluviométrie irrégulière, fertilité organique parfois faible, accès à l'eau.",
        "sol": "Sols ferrugineux tropicaux fréquents ; surveiller matière organique et érosion.",
        "cultures": ["Oignon", "Tomate", "Piment", "Gombo", "Niébé"],
        "conseils": "Rotation céréales-légumineuses, compost, collecte d'eau et maraîchage près des points d'eau."
    },
    "Fatick": {
        "zone": "Bassin arachidier / Sine-Saloum",
        "forces": "Diversité agricole, potentiel horticole et systèmes agro-sylvo-pastoraux.",
        "faiblesses": "Salinité localisée, variabilité des pluies et dégradation de certains sols.",
        "sol": "Sols ferrugineux et zones halomorphes localisées ; analyse de conductivité recommandée.",
        "cultures": ["Oignon", "Tomate", "Gombo", "Piment", "Pastèque"],
        "conseils": "Choisir les parcelles hors zones salées, améliorer la matière organique et sécuriser l'eau."
    },
    "Kaffrine": {
        "zone": "Bassin arachidier / centre",
        "forces": "Disponibilité de terres et systèmes de polyculture-élevage.",
        "faiblesses": "Stress hydrique, pluies irrégulières et fertilité à entretenir.",
        "sol": "Sols ferrugineux tropicaux ; risque de faible matière organique.",
        "cultures": ["Oignon", "Tomate", "Gombo", "Piment", "Niébé"],
        "conseils": "Compost, rotation, variétés adaptées au calendrier pluviométrique et irrigation économe."
    },
    "Kaolack": {
        "zone": "Bassin arachidier / Sine-Saloum",
        "forces": "Marchés agricoles, savoir-faire et possibilités de diversification.",
        "faiblesses": "Salinité dans certains secteurs, pression hydrique et baisse de fertilité.",
        "sol": "Sols ferrugineux et sols halomorphes localisés ; mesurer CE et pH.",
        "cultures": ["Oignon", "Tomate", "Gombo", "Piment", "Pastèque"],
        "conseils": "Diagnostic de salinité avant investissement, amendement organique et irrigation contrôlée."
    },
    "Kédougou": {
        "zone": "Sénégal oriental",
        "forces": "Pluviométrie relativement plus élevée, diversité agroécologique et potentiel horticole.",
        "faiblesses": "Enclavement, relief, accès aux intrants et éloignement des marchés.",
        "sol": "Sols tropicaux et lithosols selon relief ; profondeur et pente à vérifier.",
        "cultures": ["Tomate", "Piment", "Gombo", "Aubergine", "Chou"],
        "conseils": "Sécuriser débouchés et conservation, protéger les sols en pente et planifier le transport."
    },
    "Kolda": {
        "zone": "Haute-Casamance",
        "forces": "Potentiel pluviométrique, diversité culturale et systèmes agroforestiers.",
        "faiblesses": "Accès aux marchés, conservation, pression parasitaire et drainage local.",
        "sol": "Sols ferrallitiques/ferrugineux selon les secteurs ; surveiller acidité et drainage.",
        "cultures": ["Tomate", "Piment", "Gombo", "Aubergine", "Chou"],
        "conseils": "Compost, drainage, protection phytosanitaire intégrée et transformation locale."
    },
    "Louga": {
        "zone": "Zone sylvopastorale / Niayes nord",
        "forces": "Potentiel d'élevage, horticulture localisée dans les Niayes et accès à certains marchés.",
        "faiblesses": "Aridité, forte dépendance à l'eau et risques de salinité.",
        "sol": "Sols sableux et ferrugineux ; faible rétention hydrique possible.",
        "cultures": ["Oignon", "Tomate", "Pastèque", "Piment", "Gombo"],
        "conseils": "Goutte-à-goutte, paillage, stockage d'eau et contrôle de la salinité."
    },
    "Matam": {
        "zone": "Vallée du fleuve / zone sylvopastorale",
        "forces": "Irrigation, décrue, riziculture et cultures de contre-saison.",
        "faiblesses": "Salinité, drainage, maintenance des ouvrages et compétition pour l'eau.",
        "sol": "Sols hydromorphes et alluviaux dans la vallée ; drainage et salinité à tester.",
        "cultures": ["Oignon", "Tomate", "Gombo", "Piment", "Aubergine"],
        "conseils": "Conditionner l'expansion à la qualité de l'eau, au drainage et à la maintenance collective."
    },
    "Saint-Louis": {
        "zone": "Delta et vallée du fleuve / Niayes nord",
        "forces": "Riz irrigué, tomate industrielle, horticulture et infrastructures hydrauliques.",
        "faiblesses": "Salinité, coûts énergétiques, drainage et dépendance à l'irrigation.",
        "sol": "Sols alluviaux/hydromorphes et secteurs salés ; analyses eau-sol indispensables.",
        "cultures": ["Oignon", "Tomate", "Chou", "Piment", "Gombo"],
        "conseils": "Optimiser les calendriers, contrôler le drainage et la salinité, mutualiser pompage et maintenance."
    },
    "Sédhiou": {
        "zone": "Moyenne-Casamance",
        "forces": "Riziculture, agroforesterie, anacarde et potentiel horticole.",
        "faiblesses": "Enclavement, transformation limitée, stockage et accès aux marchés.",
        "sol": "Sols tropicaux et bas-fonds hydromorphes ; vérifier acidité et drainage.",
        "cultures": ["Tomate", "Piment", "Gombo", "Aubergine", "Chou"],
        "conseils": "Développer stockage, séchage et transformation ; adapter les cultures aux bas-fonds."
    },
    "Tambacounda": {
        "zone": "Sénégal oriental",
        "forces": "Terres agricoles, céréales, coton, fonio et élevage.",
        "faiblesses": "Distances, accès aux services, stockage et variabilité climatique.",
        "sol": "Sols ferrugineux et sols peu évolués selon les unités ; protéger contre l'érosion.",
        "cultures": ["Tomate", "Gombo", "Piment", "Oignon", "Aubergine"],
        "conseils": "Associer production et stockage, planifier les transports et restaurer la fertilité organique."
    },
    "Thiès": {
        "zone": "Niayes / bassin arachidier",
        "forces": "Diversité horticole, débouchés urbains, infrastructures et savoir-faire.",
        "faiblesses": "Pression foncière, salinité, érosion côtière et concurrence sur l'eau.",
        "sol": "Sols sableux des Niayes et unités variées ; surveiller salinité et matière organique.",
        "cultures": ["Oignon", "Tomate", "Chou", "Piment", "Laitue"],
        "conseils": "Goutte-à-goutte, paillage, rotation, protection des terres horticoles et vente contractualisée."
    },
    "Ziguinchor": {
        "zone": "Basse-Casamance",
        "forces": "Eau et pluies, riz, mangue, anacarde et maraîchage diversifié.",
        "faiblesses": "Drainage, conservation, accès aux marchés et pertes post-récolte.",
        "sol": "Sols tropicaux et bas-fonds hydromorphes ; acidité et drainage à caractériser.",
        "cultures": ["Tomate", "Piment", "Gombo", "Aubergine", "Chou"],
        "conseils": "Associer maraîchage et arboriculture, améliorer drainage, stockage et chaîne du froid."
    },
}

# Paramètres techniques et économiques INITIAUX, modifiables dans l'application.
# Rendement t/ha/cycle ; eau m3/ha/cycle ; travail jours/ha/cycle.
CROPS = {
    "Oignon":     {"yield": 25.0, "price": 350000, "variable_cost": 4200000, "water": 4500, "labor": 95, "seed_cost": 350000, "cycle": 120},
    "Tomate":     {"yield": 30.0, "price": 300000, "variable_cost": 5200000, "water": 5200, "labor": 115, "seed_cost": 450000, "cycle": 100},
    "Piment":     {"yield": 12.0, "price": 900000, "variable_cost": 3800000, "water": 4000, "labor": 100, "seed_cost": 300000, "cycle": 150},
    "Chou":       {"yield": 28.0, "price": 250000, "variable_cost": 3900000, "water": 4300, "labor": 90, "seed_cost": 250000, "cycle": 90},
    "Gombo":      {"yield": 10.0, "price": 450000, "variable_cost": 2300000, "water": 3200, "labor": 75, "seed_cost": 180000, "cycle": 80},
    "Laitue":     {"yield": 18.0, "price": 250000, "variable_cost": 2800000, "water": 3000, "labor": 85, "seed_cost": 220000, "cycle": 60},
    "Pastèque":   {"yield": 22.0, "price": 180000, "variable_cost": 2600000, "water": 3500, "labor": 60, "seed_cost": 180000, "cycle": 90},
    "Aubergine":  {"yield": 20.0, "price": 300000, "variable_cost": 3400000, "water": 4200, "labor": 90, "seed_cost": 250000, "cycle": 120},
}

# Investissements indicatifs FCFA : à remplacer par devis locaux.
INSTALLATIONS = {
    "Forage / puits équipé": 4500000,
    "Pompe et accessoires": 1200000,
    "Kit goutte-à-goutte par hectare": 1800000,
    "Bassin / réserve d'eau": 1500000,
    "Clôture et sécurisation": 900000,
    "Magasin de stockage": 2500000,
    "Abri de conditionnement": 1800000,
    "Énergie solaire de pompage": 3500000,
}

# -------------------------------------------------------------------
# OUTILS
# -------------------------------------------------------------------
def money(x):
    return f"{x:,.0f} FCFA".replace(",", " ")

def advice_from_inputs(profile, ph, ec, organic, water_quality, drainage):
    advice = [
        f"Orientation agroécologique régionale : {profile['zone']}.",
        f"Forces territoriales : {profile['forces']}",
        f"Points de vigilance : {profile['faiblesses']}",
        f"Lecture pédologique indicative : {profile['sol']}",
        f"Orientation culturale : {profile['conseils']}",
    ]
    if ph < 5.5:
        advice.append("pH acide : faire confirmer par analyse ; envisager un amendement calcaire uniquement sur recommandation agronomique.")
    elif ph > 7.8:
        advice.append("pH élevé : vérifier disponibilité du phosphore et des oligoéléments avant correction.")
    if ec > 4:
        advice.append("Conductivité élevée : risque de salinité important pour plusieurs légumes ; tester l'eau et le sol avant plantation.")
    elif ec > 2:
        advice.append("Salinité potentielle : privilégier une gestion prudente de l'irrigation et un suivi de la conductivité.")
    if organic < 1.5:
        advice.append("Matière organique faible : compost mûr, résidus végétaux et couverture du sol à envisager.")
    if water_quality == "Médiocre":
        advice.append("Eau de qualité médiocre : analyse de salinité et de contaminants avant usage agricole.")
    if drainage == "Mauvais":
        advice.append("Drainage insuffisant : corriger le drainage avant d'intensifier les cultures sensibles à l'asphyxie racinaire.")
    return advice


def optimize_plan(crops, area, water, labor, fixed_cost, installation_annual, price_factor,
                  yield_factor, water_factor, labor_factor):
    names = list(crops.keys())
    n = len(names)
    if n == 0:
        return None, "Aucune culture sélectionnée."

    # x_j = hectares attribués à la culture j.
    revenue = np.array([crops[k]["yield"] * yield_factor *
                        crops[k]["price"] * price_factor for k in names])
    varcost = np.array([crops[k]["variable_cost"] for k in names])
    water_need = np.array([crops[k]["water"] * water_factor for k in names])
    labor_need = np.array([crops[k]["labor"] * labor_factor for k in names])
    unit_margin = revenue - varcost

    def objective(x):
        # Minimisation de l'opposé de la marge totale.
        return -(unit_margin @ x)

    constraints = [
        {"type": "ineq", "fun": lambda x: area - np.sum(x)},
        {"type": "ineq", "fun": lambda x: water - water_need @ x},
        {"type": "ineq", "fun": lambda x: labor - labor_need @ x},
    ]
    x0 = np.full(n, max(area, 0.01) / n)
    result = minimize(objective, x0, method="SLSQP",
                      bounds=[(0, area) for _ in names],
                      constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-8})
    if not result.success:
        return None, result.message

    x = np.maximum(result.x, 0)
    rows = []
    for j, name in enumerate(names):
        c = crops[name]
        hectares = x[j]
        production = hectares * c["yield"] * yield_factor
        ca = production * c["price"] * price_factor
        cout = hectares * c["variable_cost"]
        rows.append({
            "Culture": name, "Surface (ha)": hectares,
            "Production (t)": production, "Chiffre d'affaires (FCFA)": ca,
            "Coûts variables (FCFA)": cout,
            "Marge avant frais fixes (FCFA)": ca - cout,
            "Eau (m³)": hectares * c["water"] * water_factor,
            "Travail (jours)": hectares * c["labor"] * labor_factor,
        })
    plan = pd.DataFrame(rows)
    total_ca = plan["Chiffre d'affaires (FCFA)"].sum()
    total_var = plan["Coûts variables (FCFA)"].sum()
    total_water = plan["Eau (m³)"].sum()
    total_labor = plan["Travail (jours)"].sum()
    net = total_ca - total_var - fixed_cost - installation_annual
    summary = {
        "ca": total_ca, "var": total_var, "fixed": fixed_cost,
        "install": installation_annual, "net": net,
        "water": total_water, "labor": total_labor,
        "area": plan["Surface (ha)"].sum(),
        "production": plan["Production (t)"].sum(),
    }
    return (plan, summary), None


# -------------------------------------------------------------------
# INTERFACE PRINCIPALE
# -------------------------------------------------------------------
st.title("🌱 AgriSénégal Intelligent")
st.markdown(
    "Outil d'aide à la décision pour le diagnostic territorial, le maraîchage, "
    "les installations agricoles, l'analyse économique et l'optimisation."
)
st.warning(
    "Les valeurs techniques et économiques initiales sont des hypothèses de travail. "
    "Elles doivent être remplacées par des analyses de sol, devis, prix de marché "
    "et rendements observés dans la commune concernée."
)

tabs = st.tabs([
    "1. Territoires", "2. Sols & diagnostic", "3. Cultures",
    "4. Installations", "5. Optimisation économique", "6. Conseils & données"
])

# Sidebar : territoire et dimensionnement
with st.sidebar:
    st.header("Localisation du projet")
    region = st.selectbox("Région", list(REGIONS.keys()))
    commune = st.text_input("Commune / village", placeholder="Saisir le nom exact")
    profile = REGIONS[region]
    st.caption(f"Zone agroécologique : {profile['zone']}")
    st.divider()
    st.header("Dimensions de l'exploitation")
    area = st.number_input("Surface disponible (ha)", min_value=0.05, max_value=10000.0,
                           value=2.0, step=0.25)
    water = st.number_input("Eau disponible par cycle (m³)", min_value=0.0,
                            value=12000.0, step=500.0)
    labor = st.number_input("Main-d'œuvre disponible (jours/cycle)", min_value=0.0,
                            value=500.0, step=25.0)
    st.caption("Les ressources saisies sont des plafonds de planification.")

with tabs[0]:
    st.subheader("Diagnostic régional et communal")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### {region}")
        st.markdown(f"**Zone agroécologique :** {profile['zone']}")
        st.markdown(f"**Forces :** {profile['forces']}")
        st.markdown(f"**Faiblesses :** {profile['faiblesses']}")
    with c2:
        st.markdown("### Commune / village")
        if commune.strip():
            st.info(f"Territoire sélectionné : **{commune.strip()}**, région de **{region}**.")
            st.write("Le diagnostic communal précis nécessite des données locales : coordonnées, unités pédologiques, accès à l'eau, marchés et statistiques agricoles.")
        else:
            st.info("Saisissez une commune dans le panneau latéral pour individualiser le projet.")
        st.markdown("**Cultures candidates :** " + ", ".join(profile["cultures"]))
    st.markdown("### Carte de localisation")
    st.caption("Carte schématique des centres régionaux. Pour une carte communale exacte, importer des limites communales officielles dans l'onglet 6.")
    coords = {
        "Dakar": (-17.45, 14.72), "Diourbel": (-16.23, 14.66), "Fatick": (-16.41, 14.34),
        "Kaffrine": (-15.55, 14.10), "Kaolack": (-16.07, 14.15), "Kédougou": (-12.18, 12.56),
        "Kolda": (-14.95, 12.89), "Louga": (-16.22, 15.62), "Matam": (-13.26, 15.66),
        "Saint-Louis": (-16.49, 16.03), "Sédhiou": (-15.56, 12.71), "Tambacounda": (-13.67, 13.77),
        "Thiès": (-16.93, 14.79), "Ziguinchor": (-16.27, 12.58)
    }
    map_df = pd.DataFrame([{"Région": r, "lon": xy[0], "lat": xy[1],
                            "Sélection": "Sélectionnée" if r == region else "Autre région"}
                           for r, xy in coords.items()])
    figmap = px.scatter_map(map_df, lat="lat", lon="lon", color="Sélection",
                            hover_name="Région", zoom=5.2,
                            center={"lat": 14.5, "lon": -14.5},
                            map_style="carto-positron", height=520)
    st.plotly_chart(figmap, use_container_width=True)

with tabs[1]:
    st.subheader("Diagnostic des sols et de l'eau")
    st.write(profile["sol"])
    a, b, c = st.columns(3)
    with a:
        ph = st.number_input("pH du sol", 3.0, 10.0, 6.5, 0.1)
    with b:
        ec = st.number_input("Conductivité électrique (dS/m)", 0.0, 20.0, 1.0, 0.1)
    with c:
        organic = st.number_input("Matière organique (%)", 0.0, 15.0, 2.0, 0.1)
    d, e = st.columns(2)
    with d:
        water_quality = st.selectbox("Qualité de l'eau", ["Bonne", "Moyenne", "Médiocre", "Non analysée"])
    with e:
        drainage = st.selectbox("Drainage de la parcelle", ["Bon", "Moyen", "Mauvais"])
    st.markdown("### Forces et faiblesses à partir des observations saisies")
    if ph < 5.5:
        st.warning("Sol potentiellement acide : analyse complémentaire recommandée.")
    elif ph > 7.8:
        st.warning("Sol alcalin : vérifier les risques de blocage de certains éléments nutritifs.")
    else:
        st.success("pH dans une plage généralement favorable à de nombreuses cultures maraîchères.")
    if ec > 4:
        st.error("Risque élevé de salinité : ne pas engager d'investissement sans diagnostic spécialisé.")
    elif ec > 2:
        st.warning("Salinité possible : vérifier la tolérance de la culture et la qualité de l'eau.")
    else:
        st.success("Conductivité saisie relativement faible ; interprétation à confirmer selon méthode et texture.")
    if organic < 1.5:
        st.warning("Matière organique faible : envisager compost mûr, couverture et rotations.")
    else:
        st.info("Matière organique déclarée : interpréter avec la texture et les analyses de laboratoire.")
    st.caption("Les seuils affichés sont des repères de pré-diagnostic, pas un classement pédologique officiel.")

with tabs[2]:
    st.subheader("Sélection des cultures maraîchères")
    st.write("Cultures recommandées comme candidates pour la région. Confirmer par calendrier cultural, variété, eau, sol et débouchés.")
    selected_crops = st.multiselect("Cultures à considérer dans l'optimisation",
                                    options=list(CROPS.keys()),
                                    default=[x for x in profile["cultures"] if x in CROPS][:4])
    crop_rows = []
    for name in selected_crops:
        c = CROPS[name]
        crop_rows.append({"Culture": name, "Rendement hypothétique (t/ha/cycle)": c["yield"],
                          "Prix hypothétique (FCFA/t)": c["price"],
                          "Coût variable (FCFA/ha)": c["variable_cost"],
                          "Eau (m³/ha/cycle)": c["water"],
                          "Travail (jours/ha)": c["labor"],
                          "Durée indicative (jours)": c["cycle"]})
    if crop_rows:
        st.dataframe(pd.DataFrame(crop_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Sélectionnez au moins une culture.")
    st.markdown("### Modifier les hypothèses")
    editable = {}
    for name in selected_crops:
        with st.expander(name):
            q1, q2, q3 = st.columns(3)
            editable[name] = dict(CROPS[name])
            editable[name]["yield"] = q1.number_input(f"Rendement {name} (t/ha)",
                min_value=0.1, value=float(CROPS[name]["yield"]), step=0.5, key=f"y_{name}")
            editable[name]["price"] = q2.number_input(f"Prix {name} (FCFA/t)",
                min_value=1, value=int(CROPS[name]["price"]), step=10000, key=f"p_{name}")
            editable[name]["variable_cost"] = q3.number_input(f"Coût variable {name} (FCFA/ha)",
                min_value=0, value=int(CROPS[name]["variable_cost"]), step=100000, key=f"c_{name}")
            q4, q5 = st.columns(2)
            editable[name]["water"] = q4.number_input(f"Eau {name} (m³/ha)",
                min_value=0, value=int(CROPS[name]["water"]), step=100, key=f"w_{name}")
            editable[name]["labor"] = q5.number_input(f"Travail {name} (jours/ha)",
                min_value=0, value=int(CROPS[name]["labor"]), step=5, key=f"l_{name}")
    st.session_state["selected_crops"] = editable

with tabs[3]:
    st.subheader("Dimensionnement des installations agricoles")
    st.write("Cochez les équipements nécessaires. Les montants sont des enveloppes indicatives à remplacer par des devis.")
    chosen_installations = []
    total_capex = 0
    cols = st.columns(2)
    for i, (name, cost) in enumerate(INSTALLATIONS.items()):
        with cols[i % 2]:
            if st.checkbox(f"{name} — {money(cost)}", value=(name in ["Pompe et accessoires", "Kit goutte-à-goutte par hectare"]),
                           key=f"inst_{i}"):
                chosen_installations.append(name)
                total_capex += cost * (area if "par hectare" in name else 1)
    life = st.number_input("Durée d'amortissement (années)", 1, 30, 7)
    annual_maintenance = st.number_input("Maintenance annuelle (FCFA)", 0, 100000000,
                                         250000, step=50000)
    annual_capex = total_capex / life if life else 0
    st.metric("Investissement initial estimé", money(total_capex))
    st.metric("Amortissement annuel linéaire", money(annual_capex))
    st.metric("Maintenance annuelle", money(annual_maintenance))
    st.session_state["installation_annual"] = annual_capex + annual_maintenance
    st.session_state["capex"] = total_capex
    st.session_state["install_names"] = chosen_installations
    if chosen_installations:
        st.dataframe(pd.DataFrame([{"Installation": n, "Coût estimé (FCFA)": INSTALLATIONS[n] * (area if "par hectare" in n else 1)}
                                   for n in chosen_installations]), use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Optimisation des coûts, revenus et bénéfices")
    st.markdown("**Objectif :** maximiser la marge nette sous contraintes de surface, d'eau et de main-d'œuvre.")
    crop_params = st.session_state.get("selected_crops", {})
    if not crop_params:
        crop_params = {n: CROPS[n] for n in profile["cultures"] if n in CROPS}
    u1, u2, u3 = st.columns(3)
    with u1:
        fixed_cost = st.number_input("Frais fixes du cycle (FCFA)", 0, 100000000,
                                     500000, step=50000)
    with u2:
        price_factor = st.slider("Facteur de prix de vente", 0.5, 1.5, 1.0, 0.05)
    with u3:
        yield_factor = st.slider("Facteur de rendement", 0.4, 1.4, 1.0, 0.05)
    v1, v2 = st.columns(2)
    with v1:
        water_factor = st.slider("Facteur de besoin en eau", 0.7, 1.5, 1.0, 0.05)
    with v2:
        labor_factor = st.slider("Facteur de besoin en travail", 0.7, 1.5, 1.0, 0.05)
    if st.button("Calculer le plan optimal", type="primary"):
        output, error = optimize_plan(crop_params, area, water, labor, fixed_cost,
                                      st.session_state.get("installation_annual", 0),
                                      price_factor, yield_factor, water_factor, labor_factor)
        if error:
            st.error(f"Optimisation non aboutie : {error}")
        else:
            plan, summary = output
            st.session_state["optimized_plan"] = plan
            st.session_state["optimized_summary"] = summary
    if "optimized_plan" in st.session_state:
        plan = st.session_state["optimized_plan"]
        summary = st.session_state["optimized_summary"]
        st.dataframe(plan.style.format({
            "Surface (ha)": "{:.3f}", "Production (t)": "{:.2f}",
            "Chiffre d'affaires (FCFA)": "{:,.0f}",
            "Coûts variables (FCFA)": "{:,.0f}",
            "Marge avant frais fixes (FCFA)": "{:,.0f}",
            "Eau (m³)": "{:,.0f}", "Travail (jours)": "{:,.1f}"
        }), use_container_width=True)
        k1, k2, k3 = st.columns(3)
        k1.metric("Chiffre d'affaires", money(summary["ca"]))
        k2.metric("Coûts variables", money(summary["var"]))
        k3.metric("Résultat net estimé", money(summary["net"]),
                  delta="Bénéfice" if summary["net"] >= 0 else "Déficit")
        st.write({
            "Surface utilisée (ha)": round(summary["area"], 3),
            "Production estimée (t)": round(summary["production"], 2),
            "Eau consommée (m³)": round(summary["water"], 1),
            "Travail mobilisé (jours)": round(summary["labor"], 1),
            "Investissement initial (FCFA)": st.session_state.get("capex", 0)
        })
        fig = px.bar(plan, x="Culture", y="Marge avant frais fixes (FCFA)",
                     color="Culture", title="Marge par culture")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Télécharger le plan optimal (CSV)",
                           plan.to_csv(index=False).encode("utf-8-sig"),
                           "plan_optimal_agricole.csv", "text/csv")
        st.caption("Le résultat net soustrait les coûts variables, frais fixes et amortissement/maintenance annuelle saisis. Il ne constitue pas un bénéfice comptable certifié.")

with tabs[5]:
    st.subheader("Conseils opérationnels et import de données communales")
    st.markdown("### Conseils adaptés au territoire et au diagnostic")
    for item in advice_from_inputs(profile, ph, ec, organic, water_quality, drainage):
        st.markdown(f"- {item}")
    st.markdown("### Modèle de données à l'échelle communale")
    st.write("Importez un CSV communal pour enrichir le diagnostic. Une ligne doit correspondre à une commune ou à une parcelle.")
    template = pd.DataFrame([{
        "region": region, "commune": commune or "A_COMPLETER",
        "latitude": "", "longitude": "", "surface_ha": "",
        "type_sol": "", "ph": "", "conductivite_ds_m": "",
        "matiere_organique_pct": "", "eau_disponible_m3": "",
        "culture": "", "production_t": "", "prix_fcfa_t": "",
        "cout_total_fcfa": "", "source": "", "annee": ""
    }])
    st.download_button("Télécharger le modèle CSV communal",
                       template.to_csv(index=False).encode("utf-8-sig"),
                       "modele_donnees_communes_senegal.csv", "text/csv")
    commune_file = st.file_uploader("Importer les données des communes", type=["csv"], key="commune_csv")
    if commune_file:
        try:
            cdf = pd.read_csv(commune_file)
            st.dataframe(cdf, use_container_width=True)
            if {"latitude", "longitude"}.issubset(cdf.columns):
                cdf["latitude"] = pd.to_numeric(cdf["latitude"], errors="coerce")
                cdf["longitude"] = pd.to_numeric(cdf["longitude"], errors="coerce")
                cdf = cdf.dropna(subset=["latitude", "longitude"])
                if not cdf.empty:
                    figc = px.scatter_map(cdf, lat="latitude", lon="longitude",
                                          color="region" if "region" in cdf.columns else None,
                                          hover_name="commune" if "commune" in cdf.columns else None,
                                          map_style="carto-positron", zoom=5,
                                          height=500, title="Localisation des communes / parcelles")
                    st.plotly_chart(figc, use_container_width=True)
            st.download_button("Exporter les données importées",
                               cdf.to_csv(index=False).encode("utf-8-sig"),
                               "donnees_communes_nettoyees.csv", "text/csv")
        except Exception as exc:
            st.error(f"Erreur de lecture du fichier communal : {exc}")

    st.markdown("### Méthode et limites")
    st.markdown("""
    - **Territorialisation :** les profils régionaux constituent une première orientation qualitative.
    - **Commune/parcelle :** les recommandations précises exigent une localisation et des données pédologiques mesurées.
    - **Optimisation :** programme linéaire continu sur les hectares par culture, avec contraintes de surface, eau et travail.
    - **Économie :** chiffre d'affaires = production × prix ; résultat estimé = chiffre d'affaires − coûts variables − frais fixes − amortissement/maintenance.
    - **Installations :** montants indicatifs, à confirmer par devis locaux et étude technique.
    - **Rendements/prix :** valeurs de départ pédagogiques ; elles ne sont pas présentées comme statistiques officielles.
    """)

st.divider()
st.caption("AgriSénégal Intelligent — prototype d'aide à la décision. Validation agronomique, pédologique et économique locale indispensable.")
