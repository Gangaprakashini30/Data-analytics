# Import necessary libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Scrape the Data
base_url = "http://books.toscrape.com/catalogue/page-{}.html"
titles = []
prices = []

for page in range(1, 6):  # Scraping first 5 pages
    print(f"Scraping page {page}...")
    url = base_url.format(page)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    books = soup.find_all("article", class_="product_pod")
    
    for book in books:
        title = book.h3.a["title"]
        price = book.find("p", class_="price_color").text
        titles.append(title)
        prices.append(price)

# Step 2: Create DataFrame
df = pd.DataFrame({"Title": titles, "Price": prices})

# Step 3: Clean the Data
df["Price"] = df["Price"].str.replace("£", "").astype(float)

# Step 4: EDA - Basic Analysis
print("\nTop 5 Entries:")
print(df.head())

print("\nSummary Statistics:")
print(df["Price"].describe())

# Step 5: Visualization
plt.figure(figsize=(10, 6))
sns.histplot(df["Price"], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Book Prices")
plt.xlabel("Price (£)")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# Top 10 most expensive books
top_10 = df.sort_values(by="Price", ascending=False).head(10)
print("\nTop 10 Most Expensive Books:")
print(top_10)

# Barplot of top 10
plt.figure(figsize=(12, 6))
sns.barplot(data=top_10, x="Price", y="Title", palette="viridis")
plt.title("Top 10 Most Expensive Books")
plt.xlabel("Price (£)")
plt.ylabel("Book Title")
plt.tight_layout()
plt.show()
