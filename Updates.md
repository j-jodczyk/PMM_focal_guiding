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