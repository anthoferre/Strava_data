from stravalib.client import Client

# --- Étape 1 : Configuration et génération de l'URL d'autorisation ---

# Remplacez ces valeurs par celles de votre application Strava
CLIENT_ID = '178290'
CLIENT_SECRET = '4cec3be900b3ae76fa0c87b9e190ff74f0f73660'

# L'URL où Strava va vous rediriger après l'autorisation
# Doit correspondre à ce que vous avez mis dans les paramètres de votre app Strava
REDIRECT_URI = 'http://localhost'

# Les permissions que votre application demande
# 'read' pour accéder à vos données publiques
# 'activity:read_all' pour accéder à toutes vos activités, même privées
# 'profile:read_all' pour votre profil
SCOPE = ['read', 'activity:read_all', 'profile:read_all']

# Créez une instance du client stravalib
client = Client()

# Générez l'URL d'autorisation
url = client.authorization_url(client_id=CLIENT_ID, redirect_uri=REDIRECT_URI, scope=SCOPE)

print(f"Étape 1 : Génération de l'URL d'autorisation.")
print(f"Veuillez copier/coller cette URL dans votre navigateur web et valider :")
print(url)
print("-" * 50)

# --- Étape 2 & 3 : Autorisation manuelle et récupération du code ---

print("Étape 2 & 3 : Autorisation sur Strava et récupération du code d'autorisation.")
print("Après avoir autorisé l'application, vous serez redirigé vers une page 'Impossible d'atteindre ce site'.")
print("C'est normal. Copiez l'URL de cette page qui commence par 'http://localhost/?state=&code=...'")
authorization_response = input("Collez l'URL complète ici : ")

# Extraire le code d'autorisation de l'URL
try:
    from urllib.parse import parse_qs, urlparse
    parsed_url = urlparse(authorization_response)
    code = parse_qs(parsed_url.query)['code'][0]
    print("Code d'autorisation récupéré :", code)
except Exception as e:
    print(f"Erreur lors de la récupération du code : {e}")
    print("Veuillez vous assurer que l'URL que vous avez collée est la bonne.")
    exit()

# --- Étape 4 : Échange du code contre un jeton ---
print("-" * 50)
print("Étape 4 : Échange du code contre un jeton d'accès.")
try:
    token_response = client.exchange_code_for_token(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        code=code
    )

    access_token = token_response['access_token']
    refresh_token = token_response['refresh_token']
    expires_at = token_response['expires_at']

    print("Jeton d'accès (access_token) généré avec succès !")
    print(f"Access Token : {access_token}")
    print(f"Refresh Token : {refresh_token}")
    print(f"Jeton expire à : {expires_at} (Unix Timestamp)")

    # Sauvegardez ces jetons.
    # Pour un usage personnel, vous pouvez les mettre dans un fichier .json ou .env
    # Pour le moment, copiez-les et collez-les dans votre script initial.

except Exception as e:
    print(f"Erreur : Impossible d'échanger le code pour un jeton. {e}")
    print("Vérifiez que le Client ID, Client Secret et le code d'autorisation sont corrects.")