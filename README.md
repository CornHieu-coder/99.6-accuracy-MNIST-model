# 99.6-accuracy-MNIST-model
Slight improvements to bump up accuracy, performance should be around 99,6%. 
It is a bit disappointing 99,9% is not achieved. Though, here are some tweaks i made to pytorch sample:
  1. Divide networks into blocks much like ResNet.
  2. Add some common layers such as Batch Normalization and Max Pooling.
  3. Switch AdaDelta to AdamW.
  4. Re-write for pytorch to look specifically for CUDA.
  5. Add some transformations to improve variability ( Note that MNIST is an already established dataset that does not have much variations and the numbers are centered. Hence, I aimed for transformations that have little to medium impact only):
       a. Random Rotation
       b. Random Affine ( only moving and re-scaling image )
       c. Elastic Distortion ( Mimicking natural hand-writing )
       Note that Elastic Distortion is very impactful in bumping up accuracy. For the sake of reference, i will put the paper that back up my idea here:         https://cognitivemedium.com/assets/rmnist/Simard.pdf
     
     
