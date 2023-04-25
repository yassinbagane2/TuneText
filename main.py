import nltk 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from songs import songs
from flask import Flask, render_template, request, url_for


# Spotify API Configuration
client_id = 'f4610b3aba6c40aca83a81466c8ec8bc'
client_secret = 'c23259874c824404abfbe7c751b6e8d7'
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


stop_words = set(stopwords.words("english"))

for song_title, song_lyrics in songs.items():
    
    songs[song_title] = "".join(char.lower() for char in song_lyrics if char.isalnum() or char.isspace())

for song_title in songs:
    # Split the string value into a list of words
    words = songs[song_title].split(" ")
    
    # Remove the words that exist in the `words_to_remove` array
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # Join the remaining words back into a string and update the dictionary value
    songs[song_title] = ' '.join(filtered_words)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(songs.values())
tfidf_array = tfidf_matrix.toarray()



app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def index():
    image_url = None
    if request.method == 'POST':
        input_lyrics = request.form['lyrics']
        # Compute the TF-IDF vector for the input lyrics
        input_tfidf = vectorizer.transform([input_lyrics]).toarray()
        # Compute the cosine similarity between the input TF-IDF vector and the song TF-IDF vectors
        cosine_similarities = cosine_similarity(input_tfidf, tfidf_array)
        # Get the index of the song with the highest cosine similarity
        max_index = cosine_similarities.argmax()
        # Get the name of the song with the highest cosine similarity
        song_title = list(songs.keys())[max_index]

        song_metadata = spotify.search(q=song_title, type='track')['tracks']['items'][0]
        song_name = song_metadata['name']
        image_url = song_metadata['album']['images'][0]['url']
        album_name = song_metadata['album']['name']
        release_date = song_metadata['album']['release_date']
        artist_name = song_metadata['artists'][0]['name']
        track_url = song_metadata['external_urls']['spotify']
        return render_template("index.html",song_name = song_name, image_url=image_url, album_name=album_name, release_date=release_date.split("-")[0], artist_name=artist_name, track_url=track_url)
    else:
        return render_template("index.html", song_name="",image_url="", album_name="", release_date="", artist_name="", track_url="")

if __name__ == "__main__":
    app.run(debug=True)
