User notes:
Previous issues have been somewhat fixed. 
Other issues found:
- The overlay only works in borderless windowed mode. It does not work in fullscreen mode. I also get a notification saying "CS2 window not found - using full screen" even though it is in borderless windowed mode/Fullscreen mode.
- Sometimes it detects objects as named as "Class" it has a blue box around it.
- The program sometimes detects other stuff such as the knife im holding, grenades, weapons as the enemy.
- The program detects teammates as enemies.
- Sometimes the overlay boxes flick probably due to the yolo model detecting different game objects.
Plans that I have acted upon: 
- Research on how to train the model. I created the dataset folder which contains train a folder full of training data and images explaining the enemies and what to detect. And the valid folder which has the same purpose as train but for validation (Images succesfuly detected by the AI after the run of train_script.py).
- Fetch possible images/videos of the game in order to give the model an idea of what to detect. (In Progress).
