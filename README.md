# InBrowserCharacterRecognition
[Train a neural net to recognize handwritten characters live in your browser!](https://asa55.github.io/InBrowserCharacterRecognition/index.html)

## This code is in beta. There are more features I intend to add for users who aren't familiar with neural nets to understand what's happening under the hood. This currently works on desktop and on mobile - please report any bugs in the issues section.

### Quick Start:
* TL;DR: Open the web app, click "cheat" one or more times, draw something in the box.
* .
* The link takes you to a simple user interface with three buttons and a canvas
* Draw anything you want in the canvas (you can train on up to 10 classes)
   * You'll see the neural net takes an initial guess - at this stage it's just a baby... It hasn't been trained at all yet so the result is based on a random initialization of the weights in the network.
* The intention is for users to draw any digit 0 to 9
* Select the digit you drew from the dropdown menu (which is only relevant for training)
   * This will automatically train "batch size 1" for a single epoch based on what you drew in the box.
   * The neural net may or may not update its guess, but you did influence the learning process by adding a training sample.
   * Behind the scenes, the image is downsampled by a factor of 4 (2 pixels in the horizontal direction and 2 in the vertical direction).
   * The result is stretched into a vector.
   * The vector has nonnegative entries. A new vector is created from this where nonzero values are mapped to 1 and the remaining entries are mapped to -1.
   * This new vector represents the image data as seen by the input to the neural net (it has a size of 512, due to the number of remaining pixels after downsampling in this case).
   * The inputs are densely connected to a single hidden layer with 100 sigmoidally activated nodes.
   * The hidden layer is densely connected to 10 linearly activated output nodes.
   * The class you chose previously generated a one-hot-encoding vector of length 10, which is compared to the current output of the neural net given the input image data.
   * The error is calculated using a flavor of cross-entropy, which is used to update the network using backpropagation (see code for the optimizer details).
   * You may have noticed clicking the "train" button kicks of a single training epoch with batch size 1. The batch size can't be changed in this demo since I don't add any of your previous images to memory for future use. This could be added easily, but that's not the point of this demo.
* Click "clear" to clear the canvas.
* Change the class, draw the number for that class, click train, repeat...
* As you continue with this process, you'll notice the neural net gets better and better at recognizing the digits you draw as you draw them.
* In this demo, you're quite literally training a primitive intelligence from scratch.
* Try different ways to make the training easier and harder for your neural net!
* Realize that the neural net is uninitialized when you first start. There is no reason to expect it to guess any numbers correctly until you train on them (at least once for each, possibly more depending on how you choose to incorporate some of the ideas below).
* This isn't a test, but some things to experiment with and think about that may be entertaining are below:
   * See how many samples of each number you need to provide before the neural net gets most numbers right most of the time.
   * Which numbers does it have the hardest time with? Can you think of any reasons why this would be the case? (hint: some numbers are more similar in appearance than others)
   * See what happens when you train one number many times in a row compared to training a different number each time. Does this help or hurt the performance of the neural net as you continue this trend?
   * What happens when you provide more samples of one number than others? Do you notice the neural net guessing this number more often? Why do you think this is the case?
