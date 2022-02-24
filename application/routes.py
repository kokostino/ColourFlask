from application import app
from flask import request
from flask import jsonify
from flask import render_template

import base64
import io


from kmedoids import get_colours


# folgendes keine Ahnung
app.config["data"] = "./info"

@app.route("/")
def index():
    return render_template("index.html", index=True )



@app.route("/", methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = io.BytesIO(decoded)
    
    prediction = get_colours(image)
    plt.clf() 
    img = io.BytesIO()
    #plt.title("la grafica por: ")
    #plt.plot(get_colours(image))
    plt.savefig(get_colours(image), format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template('index.html', imagen={'imagen': plot_url })
    #return jsonify(prediction)