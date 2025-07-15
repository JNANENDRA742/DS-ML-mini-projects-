import streamlit as st
import os
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Real Time Weather API", page_icon="🌤️")

# ------------------ SESSION STATE INIT ------------------
if "user_name" not in st.session_state:
    st.session_state.user_name = None

# ------------------ LOGIN / SIGNUP PAGE ------------------
if st.session_state.user_name is None:
    st.title("🔐 User Login")
    choice = st.radio("Do You Have an Account?", ["Login", "Signup"])
    users_file = "Data/users_data.csv"
    os.makedirs("Data", exist_ok=True)

    if choice == "Signup":
        new_user = st.text_input("Enter User Name:", placeholder="User Name :")
        new_pass = st.text_input("Enter Password:", type="password", placeholder="Enter Strong Password :")

        if st.button("Create Account"):
            new_user = new_user.strip()
            if not new_user or not new_pass:
                st.warning("Username and Password cannot be empty")
            else:
                if os.path.exists(users_file):
                    with open(users_file, "r") as f:
                        existing_users = [line.strip().split(",")[0] for line in f.readlines()]
                    if new_user in existing_users:
                        st.warning("⚠️ Username already exists. Try a different one.")
                    else:
                        with open(users_file, "a") as f:
                            f.write(f"{new_user},{new_pass}\n")
                        st.success("✅ Account created successfully. Please login.")
                else:
                    with open(users_file, "w") as f:
                        f.write(f"{new_user},{new_pass}\n")
                    st.success("✅ Account created successfully. Please login.")

    elif choice == "Login":
        name = st.text_input("Enter User Name:")
        password = st.text_input("Enter Password:", type="password")

        if st.button("Login"):
            if not name or not password:
                st.warning("Username and Password cannot be empty")
            else:
                try:
                    with open(users_file, "r") as f:
                        users = [line.strip().split(",") for line in f.readlines()]
                    if [name, password] in users:
                        st.session_state.user_name = name
                        st.success("✅ Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Try Signup first if not registered.")
                except FileNotFoundError:
                    st.error("❌ No users registered yet.")
    st.stop()

# ------------------ LOGGED IN SESSION ------------------
user_name = st.session_state.user_name
st.success(f"✅ Logged in as {user_name}")

if st.sidebar.button("🚪 Logout"):
    st.session_state.user_name = None
    st.rerun()

# ------------------ FEEDBACK SECTION ------------------
with st.sidebar.expander("📝 Submit Feedback"):
    feedback = st.text_input("Your Feedback", placeholder="How can I improve this WebApp? and What's your opinion on This WebApp?")
    if st.button("Submit Feedback"):
        if feedback.strip():
            os.makedirs("Data", exist_ok=True)
            with open("Data/feedback.csv", "a") as f:
                f.write(f"{user_name},{feedback.strip()}\n")
            st.success("✅ Feedback submitted successfully.")

# ------------------ PAGE TITLE ------------------
st.title("🌍 Real Time Weather App")
st.subheader("This App Uses the OpenWeatherMap API to Fetch Real-Time Weather Data.")
st.markdown("## You can know the Approximate Weather Condition in your Searched Area ##")

# ------------------ DEFAULT BACKGROUND ------------------
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQb0sKR5dx_Y0aodiBZ4Ls2ZDxT5JLCZhbm8Q&s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# ------------------ WEATHER CONFIG ------------------
backgrounds = {
    "clear": "https://c1.wallpaperflare.com/preview/961/236/22/sky-cloud-sunny-weather.jpg",
    "clouds": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZ2EEls9oxlehO3A5PKcxrEKlZqfWPFYK5nw&s",
    "rain": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT4QII4D4puUiL1AJfeTIRvTmpbYFGoRjeE4A&s",
    "snow": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTFebVNmBsu_s2dDLNTMTYi-W2KE1tRGfA7PA&s",
    "mist": "https://www.shutterstock.com/image-photo/landscape-heavy-foggy-road-winter-260nw-1594521517.jpg",
    "drizzle": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT4QII4D4puUiL1AJfeTIRvTmpbYFGoRjeE4A&s",
    "haze": "https://www.shutterstock.com/image-photo/landscape-heavy-foggy-road-winter-260nw-1594521517.jpg"
}

emojis = {
    "clear": "☀️",
    "clouds": "☁️",
    "rain": "🌧️",
    "snow": "❄️",
    "mist": "🌫️",
    "drizzle": "🌦️",
    "haze": "🌫️"
}

def set_background(image_url):
    bg_style = f'''
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("{image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(bg_style, unsafe_allow_html=True)

def describe_feel(temp):
    if temp < 10:
        return "❄️ Very Cold"
    elif temp < 20:
        return "🧥 Cool"
    elif temp < 30:
        return "😊 Pleasant"
    elif temp < 38:
        return "😓 Warm"
    else:
        return "🔥 Hot and Humid"

def log_search(user_name, city):
    os.makedirs("Data", exist_ok=True)
    with open("Data/search_history.csv", "a") as f:
        f.write(f"{user_name},{city}\n")

def display_history(user_name):
    st.sidebar.header("📜 Your Search History")
    try:
        with open("Data/search_history.csv") as f:
            history = [line.strip() for line in f.readlines() if line.startswith(user_name)]
            for h in reversed(history[-10:]):
                st.sidebar.write("🔹 " + h.split(",")[1])
    except FileNotFoundError:
        st.sidebar.write("No history yet.")

# ------------------ WEATHER FETCH ------------------
city = st.text_input("Enter Country, City or Village Name:")

if st.button("Get Weather"):
    if city:
        api_key = "89abce4fc97ed48195e39db449e4b6b9"
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
        geo_response = requests.get(geo_url).json()

        if geo_response:
            lat = geo_response[0]["lat"]
            lon = geo_response[0]["lon"]
            resolved_city = geo_response[0]["name"]

            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(forecast_url)

            if response.status_code == 200:
                forecast_data = response.json()
                first_forecast = forecast_data["list"][0]

                # Display Weather Details
                temp = first_forecast["main"]["temp"]
                feels = first_forecast["main"]["feels_like"]
                humidity = first_forecast["main"]["humidity"]
                wind = first_forecast["wind"]["speed"]
                condition = first_forecast["weather"][0]["main"].lower()
                description = first_forecast["weather"][0]["description"].title()
                icon = first_forecast["weather"][0]["icon"]

                emoji = emojis.get(condition, "🌈")
                bg_url = backgrounds.get(condition, backgrounds["clear"])
                set_background(bg_url)

                st.header(f"{emoji} Forecasted Weather in {resolved_city} {emoji}")
                st.image(f"http://openweathermap.org/img/wn/{icon}@2x.png")
                st.markdown(f"**Condition**: {description}")
                st.markdown(f"**Temperature**: {temp}°C")
                st.markdown(f"**Feels Like**: {feels}°C → {describe_feel(feels)}")
                st.markdown(f"**Humidity**: {humidity}%")
                st.markdown(f"**Wind Speed**: {wind} m/s")

                log_search(user_name, city.title())
                display_history(user_name)
            else:
                st.error("❌ Could not fetch forecast data. Try again.")
        else:
            st.error("⚠️ Location not found. Try another spelling or nearby place.")
