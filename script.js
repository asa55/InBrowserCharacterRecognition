// If you're reading this, please give me a few more days to clean up my mess!

var clickX = new Array();
var clickY = new Array();
var clickDrag = new Array();
var userSelectedClass = 0;
var paint;

context = document.getElementById('mycanvas').getContext("2d");

function clearCanvas() {
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, context.canvas.width, context.canvas.height);
  clickX = Array();
  clickY = Array();
  clickDrag = Array();
}

function addClick(x, y, dragging)
{
  clickX.push(x);
  clickY.push(y);
  clickDrag.push(dragging);
}

function redraw(){
  context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    
  context.strokeStyle = "#0000ff";
  context.lineJoin = "round";
  context.lineWidth = 32;
              
  for(var i=0; i < clickX.length; i++) {
    context.beginPath();
    if(clickDrag[i] && i){
      context.moveTo(clickX[i-1], clickY[i-1]);
     }else{
       context.moveTo(clickX[i]-1, clickY[i]);
     }
     context.lineTo(clickX[i], clickY[i]);
     context.closePath();
     context.stroke();
  }
}

label = [1,0,0,0,0,0,0,0,0];
function currentClass(class_, class__) {
  label = class_;
  console.log("new class selected")
  $('#classbutton').html('Class (' + class__ + ')')
}

$('#mycanvas').mousedown(function(e){
  var mouseX = e.pageX - this.offsetLeft;
  var mouseY = e.pageY - this.offsetTop;

  paint = true;
  addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop);
  redraw();
});

$('#mycanvas').mousemove(function(e){
  if(paint){
    addClick(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
    redraw();
  }
});

$('#mycanvas').mouseup(function(e){
  paint = false;
});

$('#mycanvas').mouseleave(function(e){
  paint = false;
});




var input = [];

async function getMyCurrentImageData() {
  var myData = context.getImageData(0,0,context.canvas.width,context.canvas.height);
  var i;
  var j=0;
  for (i=3; i<(context.canvas.width*context.canvas.height*4); i+=(4*32)) { //The first multiple of 4 is to access the alpha channel and the next multiplier is to downsample - but the downsample shuld be re-worked. Right now it effectively makes a pic that's tall and skinny.
    input[j] = Math.min(Math.max(myData.data[i], 0), 1)*2-1; // minimax function clamps output to 0 or 1, then the data is balanced to -1 or +1 to help the neural net learn
    j++;
  };
}



var model = tf.sequential();

async function compileModel() {
  model.add(tf.layers.dense({inputShape: input.length, units: 512,}));
  model.add(tf.layers.dense({units: 10}));
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.softmaxCrossEntropy,
  });
}

async function runModel() {
  getMyCurrentImageData();
  var inputTensor = tf.tensor2d(input, [1, input.length]);
  var labelTensor = tf.tensor2d(label, [1, label.length]);
  await model.fit(inputTensor, labelTensor);
  console.log("ran model")
}


document.addEventListener('DOMContentLoaded', getMyCurrentImageData);
document.addEventListener('DOMContentLoaded', compileModel);

myCurrentArgMax=0;
async function makePrediction() {
  getMyCurrentImageData()
  // model.predict(tf.tensor2d(input, [1, input.length])).print()
  myCurrentArgMax = ((tf.tensor((model.predict(tf.tensor2d(input, [1, input.length])).arraySync())[0])).argMax()).arraySync();
  console.log(myCurrentArgMax)
  $('#divprediction').html(myCurrentArgMax);
}