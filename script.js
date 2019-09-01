// Data I need to persist in a way that is accessible from document scope...
var clickDrag = new Array();
var clickX = new Array();
var clickY = new Array();
var context = document.getElementById('mycanvas').getContext("2d");
var input = [];
var label = [1,0,0];
var model = tf.sequential();
var myCurrentArgMax = 0;
var paint;
var currentEpoch = 0;
var userSelectedClass = 0;

// Alphabetized functions
function addClick(x, y, dragging)
{
  clickX.push(x);
  clickY.push(y);
  clickDrag.push(dragging);
}

function clearCanvas() {
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, context.canvas.width, context.canvas.height);
  clickX = Array();
  clickY = Array();
  clickDrag = Array();
  $('#divprediction').html("");
}

async function compileModel() {
  model.add(tf.layers.dense({inputShape: input.length, units: 512,}));
  model.add(tf.layers.dense({units: 15, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 3}));
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.softmaxCrossEntropy,
  });
}

function currentClass(class_) {
  label = [0,0,0];
  label[class_] = 1;
  runModel();
}

async function getMyCurrentImageData() {
  var myData = context.getImageData(0,0,context.canvas.width,context.canvas.height);
  var i;
  var j=0;
  var k;
  input_ = [];
  for (i=3; i<(context.canvas.width*context.canvas.height*4); i+=(2*32)) { //The first multiple of 4 is to access the alpha channel and the next multiplier is to downsample - but the downsample shuld be re-worked. Right now it effectively makes a pic that's tall and skinny.
    input_[j] = Math.min(Math.max(myData.data[i], 0), 1)*2-1; // minimax function clamps output to 0 or 1, then the data is balanced to -1 or +1 to help the neural net learn
    j++;
  };
  for (k=0;k<16;k++) { //maybe there is a better way to do comprehensions in js. This basically downsamples the image vertically whereas the above downsamples horizontally
    for (i=0;i<32;i++) {
      input[32*k + i] = input_[2*k*32 + i];
    }
  }
}

function getTouchPos(canvasDom, touchEvent) {
  var rect = canvasDom.getBoundingClientRect();
  return {
    x: touchEvent.touches[0].clientX - rect.left,
    y: touchEvent.touches[0].clientY - rect.top
  };
}

function incrementEpoch() {
  currentEpoch++;
  $('#divcurrepoch').html("(The neural net has been trained " + currentEpoch + " times so far.)<br><br><br>");
}

async function makePrediction() {
  getMyCurrentImageData()
  if ((tf.tensor2d(input, [1, input.length]).min().arraySync()) < 0) {
    classString = "";
    myCurrentArgMax = ((tf.tensor((model.predict(tf.tensor2d(input, [1, input.length])).arraySync())[0])).argMax()).arraySync();
    switch (myCurrentArgMax) {
      case 0: classString = 'A';
        break;
      case 1: classString = 'B';
        break;
      case 2: classString = 'C';
        break;
      default:
      classString = 'Neural Net is confused...';
    }
    $('#divprediction').html(classString);
  } else {
    $('#divprediction').html(" ");
  }
}

async function pretrain(savedInput, savedLabel) {
  var inputTensor = tf.tensor2d(savedInput, [1, savedInput.length]);
  var labelTensor = tf.tensor2d(savedLabel, [1, savedLabel.length]);
  await model.fit(inputTensor, labelTensor);
  incrementEpoch();
};

async function pretrainOnEachClass() {
  await pretrain(img0, [1,0,0]);
  await pretrain(img1, [0,1,0]);
  await pretrain(img2, [0,0,1]);
  makePrediction();
}

function redraw(){
  context.clearRect(0, 0, context.canvas.width, context.canvas.height);
  context.strokeStyle = "#0000ff";
  context.lineJoin = "round";
  context.lineWidth = 16;
              
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

async function runModel() {
  getMyCurrentImageData();
  var inputTensor = tf.tensor2d(input, [1, input.length]);
  var labelTensor = tf.tensor2d(label, [1, label.length]);
  await model.fit(inputTensor, labelTensor);
  makePrediction();
  incrementEpoch();
}


// Mouse event handling
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
  makePrediction();
});

$('#mycanvas').mouseleave(function(e){
  paint = false;
});


// Touch event handling
context.canvas.addEventListener("touchstart", function (e) {
  mousePos = getTouchPos(context.canvas, e);
  var touch = e.touches[0];
  var mouseEvent = new MouseEvent("mousedown", {
    clientX: touch.clientX,
    clientY: touch.clientY
  });
  context.canvas.dispatchEvent(mouseEvent);
}, false);
context.canvas.addEventListener("touchend", function (e) {
  var mouseEvent = new MouseEvent("mouseup", {});
  context.canvas.dispatchEvent(mouseEvent);
}, false);
context.canvas.addEventListener("touchmove", function (e) {
  var touch = e.touches[0];
  var mouseEvent = new MouseEvent("mousemove", {
    clientX: touch.clientX,
    clientY: touch.clientY
  });
  context.canvas.dispatchEvent(mouseEvent);
}, false);

document.body.addEventListener("touchstart", function (e) {
  if (e.target == context.canvas) {
    e.preventDefault();
  }
}, {passive: false});
document.body.addEventListener("touchend", function (e) {
  if (e.target == context.canvas) {
    e.preventDefault();
  }
}, {passive: false});
document.body.addEventListener("touchmove", function (e) {
  if (e.target == context.canvas) {
    e.preventDefault();
  }
}, {passive: false});


// Initialization right after DOM load
document.addEventListener('DOMContentLoaded', getMyCurrentImageData);
document.addEventListener('DOMContentLoaded', compileModel);