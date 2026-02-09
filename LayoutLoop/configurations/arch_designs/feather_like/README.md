FEATHER-Like Architecture
----------------------------
This folder contains an architecture based on the FEATHER accelerator
[here](https://arxiv.org/abs/2405.13170).

This version of FEATHER only works for GEMM, and for convolution, a different set of constraints would be needed.

Q&As:
----------------------------
1. Why is the technology different?

   Since the open-sourced energy estimation plug-ins have the most flexible support
    for the 45nm components, we changed the technology to 45nm.

2. How is FEATHER being modeled?

   The key of FEATHER accelerator is enabling flexible regrouping of available PEs to execute
   different amount of dot products with different sizes concurrently. Therefore, the key of mapping
   serach lies at how to figure out the regroup strategy of PEs for a workload. To achieve this goal, 
   for architecture, we implement FEATHER.yaml with a 1D vector of PEs to enable arbitrary grouping of PEs. 
   And then we add mapping constraint to restrict each PE to hold a AH weights locally by setting a 
   WeightsBuffer for each PE with depth=AH. Each PE performs dot product of at most size AH, which generates 
   a single output, hence we add depth=1 OutputBuffer for each PE. Further, these buffers should be constrainted
   for weights and output. And the all weights in each PE need to be reduced together, such that their reduction dimension
   need to be different. Therefore, for GEMM of MxK * KxN -> MxN, we set K=AH for WeightsBuffer.

3. Doe this design perform exactly as the FEATHER design in the paper?

   It does for regular workloads where shapes of the GEMM perfectly divides the shape of NEST.
   However, since layoutloop's mapper is not as flexible as a manually generated mapping.
   Sometimes, it does not produce the exact results as shown in the paper.
   The main limitation is that layoutloop assumes a single nested loop. While in FEATHER, each
   column of PE array could support independent nested loop.

4. How long do the layoutloop simulations take?

   Depending on your workload, the simulation takes various amount of time to finish. Generally, they should
   converge within 30 mins. You can manually stop the exploration when you see things are converging by
   pressing `ctrl + C`. They sometimes will take much longer to automaticaly stop as we set the converging cretiria to be pretty high to avoid early-stop with subooptimal mappings. Use you own
   judgement.

5. How to change the scale of FEATHER?



