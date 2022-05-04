# we are going to use the flask micro web framework
# for our web app
from flask import Flask
from flask import jsonify
from flask import request
import os
import pickle
# create the web app
# flask runs by default on port 5000
app = Flask(__name__)

# we are now ready to set up our first "route".
# route is a function that handles a request.
@app.route("/", methods=["GET"])
def index():
    #we can return content and status code
    return "<h1>Welcome to my web app!<h1>", 200

# now for the /predict endpoint
@app.route("/predict", methods=["GET"])
def predict():
    # parse the query string to get our 
    # instance attribute values from the client
    hp = int(request.args.get("HP",""))
    attack = int(request.args.get("Attack","")) #empty string is a default value
    defense = int(request.args.get("Defense",""))
    sp_attack = int(request.args.get("Sp_Atk",""))
    sp_defense = int(request.args.get("Sp_Def",""))
    speed = int(request.args.get("Speed",""))


    # TODO: fix the hard coding
    print(type(hp))
    prediction = predict_interviewed_well([hp, attack, defense, sp_attack, sp_defense, speed])
    #if anything goes wrong in this function, it will return None.

    if prediction is not None:
        #we were able to successfully make a prediction
        result = {"prediction": prediction}
        return result, 200
    else:
        return "Error making prediction", 400 # bad request- blaming the client lol
    

def predict_interviewed_well(instance):
    # we need an ML model to make a prediction for our instance
    #typically the model is trained offline and used later online
    #enter PICKLING. 
    #unpickle tree.p into header and tree
    infile = open("nb.p", "rb") #read binary
    header, nb = pickle.load(infile)
    infile.close()
    print(header, nb)
    print(instance)


    try:
        prediction = nb.predict([instance])
        return prediction
    except:
        print("error")
        return None


if __name__ == "__main__":
    # when deploying to "production"
    #goal is to get the flask app onto the web
    # we can set up and maintain our own server, or we can
    # use a cloud provider. The second is a much more viable option.
    # we will use Heroku. There are 4 ways to deploy to Heroku
    # today we will do 2.b.
    # we are going to deploy as a docker container using heroku.yml and git.
    #get the port and environment variable
    port = os.environ.get("PORT", 5001)
    app.run(debug=True, port=port, host="0.0.0.0") #TODO: turn debug off once it's deployed



