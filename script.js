// assumption - downsampling OK
// assumption - alpha channel is OK (instead of blue, which would be equally accurate but wouldn't let you change colors)
// doesn't support touchscreens yet
// needs a way to tell the user it tried to train

var clickDrag = new Array();
var clickX = new Array();
var clickY = new Array();
var context = document.getElementById('mycanvas').getContext("2d");
var input = [];
var label = [1,0,0,0,0,0,0,0,0,0];
var model = tf.sequential();
var myCurrentArgMax = 0;
var paint;
var userSelectedClass = 0;

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
}

async function compileModel() {
  model.add(tf.layers.dense({inputShape: input.length, units: 512,}));
  model.add(tf.layers.dense({units: 10}));
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.softmaxCrossEntropy,
  });
}

function currentClass(class_) {
  label = [0,0,0,0,0,0,0,0,0,0];
  label[class_] = 1;
  console.log("new class selected")
  $('#classbutton').html('Class (' + class_ + ')')
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

async function makePrediction() {
  getMyCurrentImageData()
  // model.predict(tf.tensor2d(input, [1, input.length])).print() // the item to the left console logs the nn output, but the line below lets us use argmax and print to user. I know it's a little weird to look at, but the idea is to convert to a 1d tensor so that argmax works, then convert the argmax output to a scalar int for user display purposes.
  myCurrentArgMax = ((tf.tensor((model.predict(tf.tensor2d(input, [1, input.length])).arraySync())[0])).argMax()).arraySync();
  console.log(myCurrentArgMax)
  $('#divprediction').html(myCurrentArgMax);
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

async function runModel() {
  getMyCurrentImageData();
  var inputTensor = tf.tensor2d(input, [1, input.length]);
  var labelTensor = tf.tensor2d(label, [1, label.length]);
  await model.fit(inputTensor, labelTensor);
  console.log("ran model")
  makePrediction();
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
  makePrediction();
});

$('#mycanvas').mouseleave(function(e){
  paint = false;
});


// the touch handling was inspired by: https://github.com/bencentra/canvas/blob/master/signature/signature.js

function getTouchPos(canvasDom, touchEvent) {
  var rect = canvasDom.getBoundingClientRect();
  return {
    x: touchEvent.touches[0].clientX - rect.left,
    y: touchEvent.touches[0].clientY - rect.top
  };
}

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

document.addEventListener('DOMContentLoaded', getMyCurrentImageData);
document.addEventListener('DOMContentLoaded', compileModel);