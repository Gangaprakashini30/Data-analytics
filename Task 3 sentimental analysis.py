import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Scrape Tweets
# -----------------------------
query = "ChatGPT OR OpenAI lang:en"
max_tweets = 100

tweets = []
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i >= max_tweets:
        break
    tweets.append([tweet.date, tweet.content])

df = pd.DataFrame(tweets, columns=["Date", "Tweet"])

# -----------------------------
# Step 2: Sentiment Analysis
# -----------------------------
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Tweet"].apply(get_sentiment)

# -----------------------------
# Step 3: Results Overview
# -----------------------------
print("\nSample Tweets with Sentiment:")
print(df.head())

print("\nSentiment Counts:")
print(df["Sentiment"].value_counts())

# -----------------------------
# Step 4: Visualize Sentiment
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Sentiment", palette="coolwarm")
plt.title("Sentiment Analysis of Tweets")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.show()
