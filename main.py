import pandas as pd
import re
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from langchain.tools import Tool
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
import dateparser

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in your environment.")

CSV_PATH = "reviews.csv"
os.makedirs(os.path.dirname(CSV_PATH) if os.path.dirname(CSV_PATH) else ".", exist_ok=True)

llm = ChatGroq(model="llama3-70b-8192", temperature=0.7, api_key=api_key)

if not os.path.exists(CSV_PATH):
    df = pd.DataFrame(columns=["date", "review", "sentiment"])
    df.to_csv(CSV_PATH, index=False)

def detect_sentiment(feedback: str) -> str:
    prompt = f"Classify the sentiment of the following feedback as Positive, Negative, or Neutral ONLY. No extra words:\n\n{feedback}"
    response = llm.invoke(prompt)
    sentiment = response.content.strip()
    if "positive" in sentiment.lower():
        return "Positive"
    elif "negative" in sentiment.lower():
        return "Negative"
    elif "neutral" in sentiment.lower():
        return "Neutral"
    else:
        return "Neutral"

def append_to_csv(review: str, sentiment: str, date: str = None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([{"date": date, "review": review, "sentiment": sentiment}])
    if os.path.exists(CSV_PATH):
        new_row.to_csv(CSV_PATH, mode='a', header=False, index=False)
    else:
        new_row.to_csv(CSV_PATH, mode='w', header=True, index=False)

def generate_reply(feedback: str) -> str:
    sentiment = detect_sentiment(feedback)
    prompt = f"A customer left this {sentiment} feedback:\n'{feedback}'\nWrite a short, polite, and professional response."
    response = llm.invoke(prompt)
    reply = response.content.strip()
    append_to_csv(feedback, sentiment)
    return reply

def parse_date_range(date_range_str: str):
    date_range_str = date_range_str.strip().lower()
    settings = {'PREFER_DATES_FROM': 'past', 'RELATIVE_BASE': datetime.now()}

        # Handle "today"
    if date_range_str == "today":
        today = datetime.now()
        start = today.replace(hour=0, minute=0, second=0, microsecond=0)
        end = today.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end

    # Handle "yesterday"
    if date_range_str == "yesterday":
        yesterday = datetime.now() - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end
    
    if "," in date_range_str:
        parts = date_range_str.split(",")
        if len(parts) == 2:
            start = dateparser.parse(parts[0].strip(), settings=settings)
            end = dateparser.parse(parts[1].strip(), settings=settings)
            if start and end:
                return start, end
    if " to " in date_range_str:
        parts = date_range_str.split(" to ")
        if len(parts) == 2:
            start = dateparser.parse(parts[0].strip(), settings=settings)
            end = dateparser.parse(parts[1].strip(), settings=settings)
            if start and end:
                now = datetime.now()
                if start.year < 2000:
                    start = start.replace(year=now.year)
                if end.year < 2000:
                    end = end.replace(year=now.year)
                return start, end
    if date_range_str.startswith("last"):
        import re
        match = re.search(r"last (\d+|[a-z]+) days", date_range_str)
        if match:
            num_days_str = match.group(1)
            word_to_num = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
            if num_days_str.isdigit():
                num_days = int(num_days_str)
            else:
                num_days = word_to_num.get(num_days_str, None)
            if num_days is not None:
                end = datetime.now()
                start = end - timedelta(days=num_days)
                return start, end
            
    single_date = dateparser.parse(date_range_str, settings=settings)
    if single_date:
        return single_date, single_date
    return None, None

class SentimentPlotter:
    def __init__(self):
        self.plot_counter = 1

    def safe_parse_date_range(self, date_range_str: str):
        try:
            start_date, end_date = parse_date_range(date_range_str)
            if not start_date or not end_date:
                raise ValueError("Could not parse dates")
            return start_date, end_date
        except Exception:
            return None, None

    def __call__(self, date_range_str: str):
        start, end = self.safe_parse_date_range(date_range_str)
        if not start or not end:
            return (" Sorry, I couldn’t understand that date range.\n"
                    "Try formats like:\n"
                    "- 'last 7 days'\n"
                    "- '2025-08-01 to 2025-08-10'\n"
                    "- '2025-08-01 , 2025-08-10'\n"
                    "- 'August 1 to August 7'")
        return self.plot_sentiment(start, end)

    def plot_sentiment(self, start_date: datetime, end_date: datetime):
        if not os.path.exists(CSV_PATH):
            return "No data found to plot. Please add some reviews first."
        df = pd.read_csv(CSV_PATH, parse_dates=["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if df.empty:
            return "No reviews found for the selected date range."
        
        full_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        # Prepare counts with all dates
        counts = df.groupby([df["date"].dt.date, "sentiment"]).size().unstack(fill_value=0)
        # Add missing dates with zeros
        counts = counts.reindex(full_dates.date, fill_value=0)
        counts.plot(kind="bar", figsize=(8, 5))
        plt.title("Sentiment Trends")
        plt.xlabel("Date")
        plt.ylabel("Number of Reviews")
        plt.tight_layout()
        
        filename = f"sentimentplot{self.plot_counter}.png"
        self.plot_counter += 1
        plt.savefig(filename)
        plt.close()
        return filename

plotter = SentimentPlotter()

sentiment_tool = Tool(
    name="SentimentDetector",
    func=detect_sentiment,
    description="Detect sentiment from customer feedback text."
)

reply_tool = Tool(
    name="ReplyGenerator",
    func=generate_reply,
    description="Generate polite, professional reply to customer feedback."
)

plot_tool = Tool(
    name="SentimentPlotter",
    func=plotter,
    description="Generate a sentiment trend plot for a given date range."
)

agent = initialize_agent(
    tools=[sentiment_tool, reply_tool, plot_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    print("Example 1: Detect Sentiment")
    print(agent.run("Detect sentiment for: The noodles were delicious and the staff was friendly."))

    #negative feedback
    print("\nExample 2: Generate Reply")
    feedback = "The service was very slow yesterday."
    reply = generate_reply(feedback)
    print(f"Feedback: {feedback}")
    print(f"Reply: {reply}")

    #positive feedback
    print("\nExample 2: Generate Reply")
    feedback = "The foods are mouth watering."
    reply = generate_reply(feedback)
    print(f"Feedback: {feedback}")
    print(f"Reply: {reply}")

    print("\nExample 3: Generate Sentiment Plot")
    plot_path = plotter("2025-08-01 , 2025-08-10")
    print(f"Plot saved as: {plot_path}")

    print("\nExample 3: Generate Sentiment Plot")
    plot_path = plotter("Today")
    print(f"Plot saved as: {plot_path}")    
