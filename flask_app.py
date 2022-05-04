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
    f_id = request.args.get("First_pokemon","") #empty string is a default value
    f_name = request.args.get("Name_first","")
    f_type1 = request.args.get("Type_1_first","")
    f_type2 = request.args.get("Type_2_first","")
    f_gen = request.args.get("Generation_first","")
    f_legend = request.args.get("Legendary_first","")
    s_id = request.args.get("Second_pokemon","") #empty string is a default value
    s_name = request.args.get("Name_second","")
    s_type1 = request.args.get("Type_1_second","")
    s_type2 = request.args.get("Type_2_second","")
    s_gen = request.args.get("Generation_second","")
    s_legend = request.args.get("Legendary_second","")
    
    hp = request.args.get("HP","")
    attack = request.args.get("Attack","") #empty string is a default value
    defense = request.args.get("Defense","")
    sp_attack = request.args.get("Sp_Atk","")
    sp_defense = request.args.get("Sp_Def","")
    speed = request.args.get("Speed","")


    # TODO: fix the hard coding
    prediction = predict_interviewed_well([f_id, f_name, f_type1, f_type2, f_gen, f_legend, s_id, s_name, s_type1, s_type2, s_gen, s_legend, hp, attack, defense, sp_attack, sp_defense, speed])
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

    try:
        prediction = nb.predict([instance])
        if prediction[0] ==1:
            return str(instance[0]) + ": " + instance[1]
        elif prediction[0] ==2:
            return str(instance[6]) + ": " + instance[7]
        else:
            print("error")
            return None
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
    app.run(debug=False, port=port, host="0.0.0.0") #TODO: turn debug off once it's deployed



