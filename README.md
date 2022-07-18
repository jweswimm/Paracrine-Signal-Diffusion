# Paracrine-Signal-Diffusion
The electrophysiology of a neuron is dictated by a series of highly nonlinear, coupled differential equations. When some neurons fire, they release a paracrine factor. The dynamics of paracrine factors can be roughly modeled by the diffusion equation to determine the downstream effects on other neurons. By modeling these interactions we hope to understand the role of paracrine factor dynamics in modulating physiological behaviors (e.g. sleep, muscle control, depression) 

The problem of complexity is combatted by further mathematical study and the problem of magnitude is combatted by the glorious area of parallel computing. Thank you, gaming industry, for providing us with advanced GPUs. Thousands of partial differential equations can now be solved at the same time. 

What once took us years may only take us hours. 

This repo contains a proof of concept document first in 1D with some notes on programming on the GPU (see https://github.com/jweswimm/Paracrine-Signal-Diffusion/blob/main/writeups/proofofconceptwriteup.pdf). This repo also contains the pytorch implementation of the 3d diffusion problem from nonuniform sources, along with the corresponding 3d CUDA paracrine code. Because our group is still working on this problem and in the process of publication, I can't release the entire cuda code publicly. The paracrine.cu and paracrine.cuh codes should give a decent understanding of the type of code I am writing. 

I will update this README using Pandoc to convert the various LaTeX documents I've made in the last year into markdown for easier explanations and more visual examples. Stay tuned!
