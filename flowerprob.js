
// Training set, [ length, width, color(0-blue and 1-red) 

var dataB1 = [2,  1, 0];
var dataB2 = [3,  1, 0];
var dataB3 = [2, .5, 0];
var dataB4 = [1,  1, 0];

var dataR1 = [3,   1.5, 1];
var dataR2 = [3.5, .5,  1];
var dataR3 = [4,   1.5, 1];
var dataR4 = [5.5, 1,   1];

// unknown type (data we want to find)

var dataU = [4.5, 1];

var all_points = [dataB1, dataB2, dataB3, dataB4, dataR1, dataR2, dataR3, dataR4];

function sigmoid(x) {
  return 1/(1 + Math.exp(-x));
}

// training 

function train() {
  let w1 = Math.random() * .2 - .1;
  let w2 = Math.random() * .2 - .1;
  let b = Math.random() * .2 - .1;
  let learning_rate = 0.2;
  for(let iter = 0; iter < 50000; iter++){
    //pick a random point
    let random_idx = Math.floor(Math.random() * all_points.length);
    let point = all_points[random_idx];
    let target = point[2]; // target stored in 3rd item in points

    // feed forward
    let z = w1 * point[0] + w2 * point[1] + b;
    let pred = sigmoid(z);

    // comparing the model prediction with the target
    let cost = (pred - target) ** 2;

    // find the slope of the cost by each parameter w1, w2, b
    // derivative of { cost } in respect to { pred } according to the chain rule
    // d cost / d prediction
    let dcost_dpred = 2 * (pred - target);

    // finding the derivative of { pred } in respect to { z }.
    // being pred = sigmoid, derivative of { sigmoid } in respect of { z }.
    // derivative of sigmoid can be written using more sigmoids!! 
    // d/dz sigmoid(z) = sigmoid(z) * (1 - sigmoid(z))
    let dpred_dz = sigmoid(z) * (1 - sigmoid(z));

    //derivatives of { z } in respect to { w1 }, { w2 } and { b }
    let dz_dw1 = point[0];
    let dz_dw2 = point[1];
    let dz_db = 1;

    // getting the partial derivatives using chain rule
    // bringing how the cost changes through each function, first through cost, then through the sigmoid
    // and finally whatever is multiplying our parameter of interest becomes the last part
    
    // derivative of { cost } in respect to { w1 }
    let dcost_dw1 = dcost_dpred * dpred_dz * dz_dw1;

    // derivative of { cost } in respect to { w2 }
    let dcost_dw2 = dcost_dpred * dpred_dz * dz_dw2;

    // derivative of { cost } in respect to { b }
    let dcost_db = dcost_dpred * dpred_dz * dz_db;

    // updating the parameters
    w1 -= learning_rate * dcost_dw1;
    w2 -= learning_rate * dcost_dw2;
    b -= learning_rate * dcost_db;
  }

  // returning w1, w2, b as an object
  return {w1: w1, w2: w2, b: b};
}

// getting the updated w1, w2 and b parameters as properties of realData object
const realData = train();
// reconstructing the z variable with the data from dataU, the array that contains length and width of the flower which we want to find the color
let realZ = dataU[0] * realData.w1 + dataU[1] * realData.w2 + realData.b;
// calling the sigmoid function to get a binary value for the program to predict the color easily
let realPred = Math.round(sigmoid(realZ));
// writing the prediction to the console as an output of the program
if(realPred === 1){
  console.log("Seems like a red flower!!");
} else {
  console.log("Seems like a blue flower!!");
}