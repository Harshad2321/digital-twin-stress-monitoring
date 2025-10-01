import requests
import random
import datetime
import time
import csv

form_url = "https://docs.google.com/forms/d/e/1FAIpQLSePfTZDlV0jzJ-f91kNSqZrCWcgr9EJ5Zt2wQyq50pwDC-h_A/formResponse"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

# Date range
start_date = datetime.date(2025, 8, 23)
end_date   = datetime.date(2025, 10, 29)
delta_days = (end_date - start_date).days

def random_date():
    d = start_date + datetime.timedelta(days=random.randint(0, delta_days))
    return d.strftime("%Y-%m-%d")

def maybe_blank(val):
    return val if random.random() > 0.2 else ""  # 20% missing

def generate_response():
    return {
        "entry.889555621": random_date(),  # Date
        "entry.867312109": str(random.randint(18, 60)),       # Age
        "entry.1417075632": str(random.randint(40, 100)),     # Body Weight
        "entry.1006204720": random.choice(["Male","Female","Prefer not to say"]), # Gender
        "entry.530769567": maybe_blank(str(random.randint(60, 120))),  # HR
        "entry.2023618169": maybe_blank(str(random.randint(90, 100))), # SpO2
        "entry.187777565": maybe_blank(random.choice(["Poor","Average","Good"])), 
        "entry.1043549710": maybe_blank(str(random.randint(3, 10))), 
        "entry.830209473": maybe_blank(random.choice(["Highly Active","Sedentary","Active","NAN"])),
        "entry.1073239940": maybe_blank(str(random.randint(0, 8))),    
        "entry.1587150136": maybe_blank(random.choice(["Regular","Irregular"])),
        "entry.88799151": maybe_blank(random.choice(["Yes","No"])),    
        "entry.588594682": maybe_blank(str(random.randint(1000, 15000))),
        "entry.1363132591": maybe_blank(str(random.randint(1, 10))),   
        "entry.772472502": maybe_blank(random.choice(["Student","Professional","Retired","Other"])) 
    }

def submit_responses(n=1500):
    with open("fast_responses.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Date","Age","Weight","Gender","HR","SpO2","SleepQuality","SleepDuration",
            "Activity","ScreenTime","MealRegularity","SleepConsistency",
            "Steps","Stress","Role"
        ])

        for i in range(n):
            data = generate_response()
            writer.writerow(data.values())
            try:
                resp = requests.post(form_url, headers=headers, data=data, timeout=6)
                if resp.status_code == 200:
                    print(f"✅ {i+1}/{n}")
                else:
                    print(f"⚠ {i+1}: Status {resp.status_code}")
            except Exception as e:
                print(f"❌ {i+1} error: {e}")

            # ⚡ much faster: 0.05–0.15 sec delay
            time.sleep(random.uniform(0.05, 0.15))

if __name__ == "__main__":
    submit_responses(2000)
