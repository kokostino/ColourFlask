from application import app

@app.route("/")
def index():
    return 'I need a vacation somewhere sunny'