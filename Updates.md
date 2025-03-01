## 27.01
Positives:
- can run program without crashing on -nan values
    - I'm deleting components which have close to 0 contribution
- merging and splitting seems to work fine

Still need fix/investigation:
- minimum and maxium number of components is not respected
- everything behind blocking lock - bad for performance (~4.5 hour render)
    - changing number of components - maybe should have fixed number but "active" and "inactive" like vmm_robus_...
- still not counting in the "contribution" factor (everything with contribution 1)
- doesn't look good at all (nothing learnt?) - collapsed to one component
![alt text](update_imgs/modern_hall_gmm_visualization_27.01.png)
![alt text](update_imgs/modern-hall_27.01.png)

## Next update: add date
- Doing GMM relearning at every loop does not make sense
- Training iterations :) - finally understand why and implemented here
### Little thoughts:
- THE RENDERING DOES NOT STOP!!!
- how the heck are the other solutions so much faster - where is my slow point
- I think I might be sabotaging myself with the size of the current tree
- do I count the whole line (both directions till the end)?
### TASKS TO DO (currenlty):
- rendering should stop
- timing investigations:
    - how much splatting takes - check
    - how much sampling takes - check
    - how does this compare to the vanila sampler - check
- training stopping cryterion (classing GMM)
- merging cryterion - should allow more merging
- the tree in the next rendering should be more granual
    - investigate adaptivness of the tree
- make sure that we explore the whole line that the ray is on
- investigate the ordering of the GMMs
### Let's sum up how it looks:
- in visualization notebooks
