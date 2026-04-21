from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

training_data = [

# 🛒 Groceries (15)
("bigbasket groceries", "Groceries"),
("dmart shopping", "Groceries"),
("vegetables market", "Groceries"),
("milk bread eggs", "Groceries"),
("grocery store purchase", "Groceries"),
("supermarket bill", "Groceries"),
("fruits and vegetables", "Groceries"),
("weekly grocery shopping", "Groceries"),
("kirana store items", "Groceries"),
("rice dal oil", "Groceries"),
("fresh produce shopping", "Groceries"),
("paneer milk curd", "Groceries"),
("household groceries", "Groceries"),
("local market shopping", "Groceries"),
("daily essentials purchase", "Groceries"),

# 🚕 Transport (15)
("uber ride", "Transport"),
("ola cab", "Transport"),
("auto fare", "Transport"),
("auto rickshaw charges", "Transport"),
("bus ticket", "Transport"),
("train ticket", "Transport"),
("metro recharge", "Transport"),
("petrol pump refill", "Transport"),
("diesel refill", "Transport"),
("taxi fare", "Transport"),
("cab to airport", "Transport"),
("fuel expenses", "Transport"),
("bike petrol", "Transport"),
("commute charges", "Transport"),
("local travel expense", "Transport"),

# 🍔 Dining (15)
("zomato order", "Dining"),
("swiggy food", "Dining"),
("restaurant dinner", "Dining"),
("lunch at cafe", "Dining"),
("starbucks coffee", "Dining"),
("mcdonalds burger", "Dining"),
("pizza hut order", "Dining"),
("dominos pizza", "Dining"),
("kfc chicken", "Dining"),
("evening snacks", "Dining"),
("chai and samosa", "Dining"),
("breakfast outside", "Dining"),
("dinner bill", "Dining"),
("food delivery", "Dining"),
("cafe cappuccino", "Dining"),

# 💡 Utilities (15)
("electricity bill", "Utilities"),
("water bill", "Utilities"),
("internet recharge", "Utilities"),
("wifi bill", "Utilities"),
("mobile recharge", "Utilities"),
("phone bill", "Utilities"),
("gas cylinder refill", "Utilities"),
("utility payment", "Utilities"),
("broadband bill", "Utilities"),
("electric bill payment", "Utilities"),
("postpaid mobile bill", "Utilities"),
("house utilities", "Utilities"),
("internet bill", "Utilities"),
("monthly utility charges", "Utilities"),
("dth recharge", "Utilities"),

# 🎬 Entertainment (15)
("netflix subscription", "Entertainment"),
("spotify premium", "Entertainment"),
("movie ticket", "Entertainment"),
("cinema booking", "Entertainment"),
("gaming purchase", "Entertainment"),
("amazon prime subscription", "Entertainment"),
("youtube premium", "Entertainment"),
("concert ticket", "Entertainment"),
("ott subscription", "Entertainment"),
("gaming console purchase", "Entertainment"),
("music subscription", "Entertainment"),
("online streaming", "Entertainment"),
("theatre ticket", "Entertainment"),
("video game purchase", "Entertainment"),
("entertainment expense", "Entertainment"),

# 🏠 Housing (10)
("house rent", "Housing"),
("monthly rent", "Housing"),
("apartment rent", "Housing"),
("home loan emi", "Housing"),
("mortgage payment", "Housing"),
("rent payment", "Housing"),
("flat rent", "Housing"),
("housing expense", "Housing"),
("lease payment", "Housing"),
("rent transfer", "Housing"),

# 🛍️ Shopping (10)
("amazon order", "Shopping"),
("flipkart purchase", "Shopping"),
("clothes shopping", "Shopping"),
("mall shopping", "Shopping"),
("online shopping", "Shopping"),
("fashion purchase", "Shopping"),
("electronics purchase", "Shopping"),
("new shoes", "Shopping"),
("shopping bill", "Shopping"),
("accessories purchase", "Shopping"),

# 🏥 Healthcare (5)
("doctor visit", "Healthcare"),
("medicine purchase", "Healthcare"),
("pharmacy bill", "Healthcare"),
("hospital charges", "Healthcare"),
("medical test", "Healthcare"),

# 📚 Education (5)
("course fee", "Education"),
("tuition fee", "Education"),
("book purchase", "Education"),
("online course", "Education"),
("exam fee", "Education"),
]

x_train, y_train = zip(*training_data)

# ✅ FIXED PIPELINE
ai_model = make_pipeline(TfidfVectorizer(ngram_range=(1,2)), MultinomialNB())
ai_model.fit(x_train, y_train)

expenses_db = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/add', methods=['POST'])
def add_expense():
    data = request.json
    description = data.get('description', '')
    amount = data.get('amount', 0)

    predicted_category = ai_model.predict([description.lower()])[0]

    new_expense = {
        "id": len(expenses_db) + 1,
        "description": description,
        "amount": float(amount),
        "category": predicted_category
    }

    expenses_db.append(new_expense)

    return jsonify(new_expense)

@app.route('/api/expenses', methods=['GET'])
def get_expenses():
    return jsonify(expenses_db)

if __name__ == '__main__':
    app.run(debug=True)