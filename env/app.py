from flask import Flask, render_template, request
import pickle
import numpy as np

 

popular_hotels = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
hotels = pickle.load(open('hotels.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

#start
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

hotel_df = pd.DataFrame(hotels)
amenities = hotel_df["amenities"].values
tfidf = TfidfVectorizer()
features = tfidf.fit_transform(amenities)
query_text = "'Pool', 'Restaurant', 'Fitness Centre with Gym / Workout Room', 'Spa', 'Room service', 'Bar/Lounge', 'Banquet Room', 'Breakfast Available', 'Business Centre with Internet Access', 'Concierge', 'Conference Facilities', 'Dry Cleaning', 'Heated pool', 'Hot Tub', 'Indoor pool', 'Laundry Service', 'Meeting rooms', 'Multilingual Staff', 'Non-smoking hotel', 'Paid Internet', 'Paid Wifi', 'Public Wifi', 'Wheelchair Access', 'Family Rooms', 'Non-smoking rooms', 'Suites'"
query_vector = tfidf.transform([query_text])
similarities = cosine_similarity(query_vector, features)
hotels['similarity scores'] = similarities[0]
most_similar = hotels['similarity scores'].max()
most_similar_hotel = hotels[hotels['similarity scores'] == most_similar]
#end

app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html', 
                            hotel_name = list(popular_hotels['hotel_name'].values),
                            avg_rating = list(popular_hotels['avg_rating'].values),
                            hotel_experience = list(popular_hotels['hotel_experience'].values),
                            address = list(popular_hotels['address'].values),
                            country = list(popular_hotels['country'].values)
                            )


@app.route('/Recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_hotels',methods = ['POST'])
def recommend():
    user_input = request.form.get('user_input')
    index = np.where(pt.index== user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    
    data = []
    for i in similar_items:
        item = []
        temp_df = hotels[hotels['hotel_name'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('hotel_name')['hotel_name'].values))
        item.extend(list(temp_df.drop_duplicates('hotel_name')['hotel_rating'].values))
        item.extend(list(temp_df.drop_duplicates('hotel_name')['hotel_experience'].values))
        item.extend(list(temp_df.drop_duplicates('hotel_name')['address'].values))
        item.extend(list(temp_df.drop_duplicates('hotel_name')['country'].values))
        
        data.append(item)


    return render_template('recommend.html',data=data)


if __name__ == "__main__":
    app.run(debug = True)
