# Parallel-Canny-edge-detection
2023-t1-finalproject-parallel-canny created by GitHub Classroom

What is this Project?
This Project is an attempt to implement Canny's algorithm in Rust from scratch and using 
parallelism to make it faster

Canny's Algorithm explained
the algorithm is divided into five main stages

- Gussian Blur -> To remove the noise, which would avoid detecting faulty edge, 
using Spatial convolution 

- Sobel Operator -> A 2D differiential (in horizontal and vertical direction) 
approximation of to calculate the gradient of the images and also directional gradient 
to indicate the orientation of the edges, using Spatial convolution

- Non Maxima Suppression -> Due to the fact that Sobel operator give out a bleed-style edges, 
non_max will suppress the weaker pixels along those edges and maintain only a thin line
(with normalization after)

- Double Thresholding and Edge Track by Hysteresis -> bascially catagorized edges into Strong Weak and 
Non Relevent, and The Edge Track Algorithm will connect the weak and strong edges together if 
only they're neighbours.


How do I appoarch this topic?

Decision on what to implement 

Since a big part of Canny's algorithm revolves around Convolution, a big part of it was basically
spent on researching the possible way Canny's algorithm can be adapted and improved(through Rust) 
in a more traditional way. However, the more time I spend on it, the more I realized the traditional 
way of parallelizing is probably the most efficient way out of all.

From what I've collected from researching there are 2 type of Convolution: Spatial and Frequential.
Spatial basically means the matrix( or kernel) multiply element-wise and put the sum of of the matrix's 
element into each pixels all over the images. This of course will be heavily benefited from parallelism.

The other one is Frequential which uses Discrete Fourier Transform to get the frequency in 2D and 
interpreted back into a readable image through Inverse DFT. This is also great for parallelism as well. 
However, once try we to understand both Convolution Work-efficienet wise, they would be 
approximately O(4*N^2*log(N)) for Frequency and O(N^2*K^2) for Spatial, meaning that would only be 
efficient when the kernels are large, of which Canny's algorithm doesnt use.

Time Complexity of Convolution from 
https://s18798.pcdn.co/videolab/wp-content/uploads/sites/10258/2021/02/Convolution.pdf

Therefore, I choose to do Spatial Convolution as not only it would have been more readable 
but more easy to manipulate the picture while at the same time mantain the same level of 
the efficiency of the program


Benchmark: M1 MacBook Air 8RAM
Not surpisingly, the parallel version of Canny's perform way better than the serial version

For 400x600 picture

Parallel: 0.75 - 1.0 seconds
Serial: 1.5 - 1.7 seconds

However, because both Gussian blur kernel can be separate into 
2 x 1D convolution
I'm interested in how it would affected the runtime by just saving a bit more arithemic work
Parallel: 0.8 - 1.0 seconds
Serial:  1.3+ seconds (which very surprised for me)

Extra:
I also try the Arc and Mutex operation when I didnt know yet how to
do data-parallel on the image in Rust(as Rust has many kinds of 
borrowing rule) to see how overhead cost really looks like

"Concurrent" Sobel + the Rest  in parallel: ~ 1.3+ seconds



What have I learn from this project and what is interesting about it?

This is a lot I have gained from this. After this project, I feel like I know 
a lot more about the image library, or at least an intuition about it. 

I'm surpised by how powerful convolution are by how it approximates such a complex 
function yet make it looks so simple. The fact that image filter can 
be done through DFT from the research also surpise me. Maybe in the future
I may have to deal with a very large kernel which I would love to see its 
blazing fast performance.







